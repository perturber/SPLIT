#general imports
import numpy as np
import cupy as cp
import json
import os
import emcee
import multiprocessing as mp

#eryn imports
from eryn.ensemble import EnsembleSampler
from eryn.state import  State
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.moves import StretchMove
from eryn.backends import HDFBackend

#few imports
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux
from few.utils.constants import YRSID_SI
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
kerr_traj = EMRIInspiral(func=KerrEccEqFlux)

#responsewrapper imports
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens

#utility tools from StableEMRIFisher
from stableemrifisher.utils import tukey, generate_PSD, SNRcalc

from .moves import SequentialAdaptiveBlockedGibbsGaussianMove, BlockedStretchMove
from .diagnostics import update_diagnostic_plots
from .priors import MarkovStudenttPrior
from .utils import compute_rhat

use_gpu = True

if not use_gpu:
    import few
    cfg_set = few.get_config_setter(reset=True)
    cfg_set.enable_backends("cpu")
    cfg_set.set_log_level("info")
    force_backend = 'cpu'
else:
    force_backend = 'gpu'

# Global variables for the worker processes
worker_d_fft = None
worker_PSD = None
worker_freq_mask = None
worker_resp_blocks = None
worker_gpu_id = None

def init_worker(d_fft_cpu, PSD_cpu, freq_mask_cpu, Tobs_block_padded, dt, tdi_kwargs_base, Nblocks, slice_length, index_lambda, index_beta, t_buffer):
    """
    Initializes the completely isolated environment for each multiprocessing worker.
    Each worker gets its own GPU context, its own copy of the static data on that GPU, and its own set of ResponseWrapper instances built natively on that GPU.
    """
    global worker_gpu_id, worker_d_fft, worker_PSD, worker_freq_mask, worker_resp_blocks
    
    # 1. Dynamically assign a GPU based on the worker's internal ID
    num_gpus = cp.cuda.runtime.getDeviceCount()
    worker_id = mp.current_process()._identity[0] 
    worker_gpu_id = (worker_id - 1) % num_gpus # Round-robin assignment (e.g., 0, 1, 2, 3, 0, 1...)
    
    cp.cuda.Device(worker_gpu_id).use()
    
    # 2. Transfer the static data natively to THIS GPU
    # static data computed on CPU in the main process once and passed as arguments to avoid any GPU memory sharing issues.
    worker_d_fft = cp.array(d_fft_cpu)
    worker_PSD = cp.array(PSD_cpu)
    worker_freq_mask = cp.array(freq_mask_cpu)
    
    # 3. Initialize the few waveform generator ON THIS GPU
    # this is for the block-specific calculations.
    inspiral_kwargs_worker = dict(buffer_length=int(1e3))
    sum_kwargs_worker = dict(pad_output=True)
    worker_wave_gen = GenerateEMRIWaveform(
        FastKerrEccentricEquatorialFlux,
        inspiral_kwargs=inspiral_kwargs_worker,
        sum_kwargs=sum_kwargs_worker,
        return_list=False
    )
    
    # 4. Build the 10 ResponseWrappers natively on THIS GPU
    worker_resp_blocks = []
    for i in range(Nblocks):
        t_shift_seconds = i * slice_length * dt
        
        tdi_kwargs_block = tdi_kwargs_base.copy()
        tdi_kwargs_block["orbits"] = EqualArmlengthOrbits()
        
        resp_block = ResponseWrapper(
            worker_wave_gen,
            Tobs=Tobs_block_padded,
            dt=dt,
            index_lambda=index_lambda,
            index_beta=index_beta,
            t0=t_shift_seconds,
            t_buffer=t_buffer,
            flip_hx=True,
            is_ecliptic_latitude=False,
            remove_garbage=True, 
            **tdi_kwargs_block,
        )
        worker_resp_blocks.append(resp_block)

def window_gen_block_worker(*pars, resp_instance, T, dt, slice_length, alpha=0.05):
    """Generates padded waveform, slices valid middle block, applies Tukey window."""
    a = resp_instance(*pars, T=T, dt=dt)
    a = cp.array(a)
    
    if a.ndim > 1:
        a_valid = a[:, :slice_length].copy()
    else:
        a_valid = a[:slice_length].copy()

    block_window = cp.array(tukey(slice_length, alpha=alpha, use_gpu=True))
    a_valid *= block_window

    return a_valid

def log_like_semicoherent(pars_in, Tobs_block_padded, dt, df, slice_length, alpha_block,
                          len_pars_names, indices_ev_in, indices_static_in,
                          indices_ev_fixed, indices_static_fixed,
                          value_fixed_static, value_fixed_ev, nu=10):
    
    global worker_gpu_id, worker_d_fft, worker_PSD, worker_freq_mask, worker_resp_blocks
    
    # Enforce GPU context for this specific worker
    cp.cuda.Device(worker_gpu_id).use()
    xp = cp

    # Robust extraction for single-walker evaluation
    evolving = np.atleast_2d(pars_in[0]) # Ensures shape is (Nblocks, ndim_ev)
    static = np.atleast_1d(pars_in[1]).flatten() # Ensures shape is strictly 1D (ndim_static,)

    Nblocks = evolving.shape[0]
    s_blocks = []

    for i in range(Nblocks):
        pars_block = np.zeros(len_pars_names)
        pars_block[indices_ev_in] = evolving[i, :]
        
        # THE FIX: static is 1D, so we assign it directly without 2D slicing
        pars_block[indices_static_in] = static
        
        pars_block[indices_static_fixed] = value_fixed_static
        pars_block[indices_ev_fixed] = value_fixed_ev[i, :]

        s_i = window_gen_block_worker(
            *list(pars_block), 
            resp_instance=worker_resp_blocks[i],
            T=Tobs_block_padded, 
            dt=dt, 
            slice_length=slice_length,
            alpha=alpha_block
        )
        
        s_blocks.append(xp.atleast_2d(xp.asarray(s_i)))

    s_xp = xp.stack(s_blocks, axis=1)
    s_fft_masked = (dt * xp.fft.rfft(s_xp, axis=-1)[..., 1:])[..., worker_freq_mask]

    # Use isolated global data arrays
    res_tilde = worker_d_fft - s_fft_masked
    bin_inner_prod = 4 * df * (res_tilde.real**2 + res_tilde.imag**2) / worker_PSD
    
    kth_term = ((nu+2)/2) * xp.log(1 + (1/nu) * bin_inner_prod)
    summed_term = xp.sum(kth_term)

    return -float(summed_term.get())

class SPLIT:
    """
    Sliced Posteriors for Long-Inspiral Trajectories (SPLIT).

    An orchestrator class designed to manage semi-coherent Bayesian parameter 
    estimation for Extreme-Mass-Ratio Inspirals (EMRIs) across multi-GPU hardware. 
    It seamlessly handles the loading of user configurations, the generation and 
    slicing of injected waveform data (including custom/non-Kerr waveforms), prior 
    boundary construction, and the execution of the Eryn-based sampler.

    Some choices, such as the Ensemble Sampler moves are currently baked in.
    """
    def __init__(self, emri_config_path, sample_config_path, out_dir, custom_injection_func=None, additional_injection_args=None):

        """
        Initializes the SPLIT pipeline by loading JSON configuration files, mapping 
        parameters, and preparing the compute environment.

        Parameters:
            emri_config_path (str): Path to the JSON file containing the physical EMRI 
                parameters, desired SNR, observation time, and slicing settings (Nblocks, fmin/max).
            sample_config_path (str): Path to the JSON file containing the Eryn MCMC settings 
                (e.g., nwalkers, nsteps, evolving vs. static parameters, backend filenames).
            out_dir (str): Root directory where all output HDF5 chains and diagnostic 
                plots will be saved.
            custom_injection_func (callable, optional): A custom FEW waveform generator 
                for injecting modified or environmentally dirtied signals. If None, 
                defaults to the standard vacuum FastKerrEccentricEquatorialFlux.
            additional_injection_args (list, optional): Additional physical parameters 
                required by the custom_injection_func beyond the standard 14 Kerr parameters.
        """

        #LOAD JSON configurations
        with open(emri_config_path, 'r') as f:
            self.emri = json.load(f)

        with open(sample_config_path, 'r') as f:
            self.samp = json.load(f)

        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.use_gpu = True
        self.xp = cp if self.use_gpu else np
        
        self.channels = [A1TDISens,E1TDISens]
        self.noise_kwargs = [{"sens_fn": c} for c in self.channels]

        self.all_param_names = ["m1", "m2", "a", "p0", "e0", "xI0", "dist", "qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]
        self.true_pars = [self.emri.get(k, 0.0) for k in self.all_param_names]

        self.custom_injection_func = custom_injection_func
        self.add_args = additional_injection_args if additional_injection_args is not None else []

    def generate_injection_data(self):
        """
        Generates the target waveform, scales it to the desired SNR, and processes it 
        for the semi-coherent likelihood.

        This method handles:
        1. Injecting either standard Kerr or custom modified waveforms.
        2. Windowing the full dataset to compute an accurate 1-year Network SNR.
        3. Scaling the luminosity distance to exactly match the requested SNR.
        4. Slicing the raw data into independent time blocks.
        5. Applying localized Tukey windows to each block to prevent spectral leakage.
        6. Computing the frequency-domain FFTs and masked Noise PSDs for the workers.
        """
        
        print("Generating waveform and scaling SNR...")

        base_func = self.custom_injection_func if self.custom_injection_func is not None else FastKerrEccentricEquatorialFlux

        wave_gen = GenerateEMRIWaveform(
            base_func,
            inspiral_kwargs=dict(buffer_length=int(1e3)),
            sum_kwargs=dict(pad_output=True),
            return_list=False
        )

        self.t0 = 0.0 #initial time for response calculation
        self.t_buffer = 10000.0 #time for garbage orbit removal
    
        self.order = 25 
        self.tdi_gen = "1st generation" 
        self.index_lambda = 8 
        self.index_beta = 7 

        resp_gen = ResponseWrapper(
            wave_gen, Tobs=self.emri['T'], dt=self.emri['dt'],
            index_lambda=self.index_lambda, index_beta=self.index_beta,
            t_buffer=self.t_buffer, t0=self.t0, flip_hx=True,
            is_ecliptic_latitude=False, remove_garbage=True,
            orbits=EqualArmlengthOrbits(), order=self.order, tdi=self.tdi_gen,
            tdi_chan="AE"
        )

        #correctly windowed data to accurately calculate 1-year SNR
        d_windowed = self.xp.array(resp_gen(*self.true_pars, *self.add_args, T=self.emri['T'], dt=self.emri['dt']))
        N_finite = d_windowed.shape[-1]
        finite_window = self.xp.array(tukey(N_finite, alpha=self.emri['alpha_block'], use_gpu=self.use_gpu))
        d_windowed *= finite_window

        PSD_full = generate_PSD(
            d_windowed, dt=self.emri['dt'], 
            noise_PSD=get_sensitivity, 
            channels=self.channels, 
            noise_kwargs=self.noise_kwargs, 
            use_gpu=self.use_gpu
        )
        current_SNR = SNRcalc(d_windowed, PSD_full, self.emri['dt'], use_gpu=self.use_gpu)

        # distance scaling
        self.emri['dist'] = self.emri['dist'] * current_SNR / self.emri['desired_SNR']
        self.true_pars = [self.emri.get(k, 0.0) for k in self.all_param_names]

        # Generate raw data for slicing
        d_raw = self.xp.atleast_2d(self.xp.array(resp_gen(*self.true_pars, *self.add_args, T=self.emri['T'], dt=self.emri['dt'])))

        n = len(d_raw[0])
        self.slice_length = int(np.ceil(n/self.emri['Nblocks']))
        pad_size = (self.emri['Nblocks'] * self.slice_length) - n
        
        pad_width = [(0,0)] * (d_raw.ndim-1) + [(0, pad_size)]
        d_padded = self.xp.pad(d_raw, pad_width, mode='constant')
        self.d_slices = d_padded.reshape(d_raw.shape[0], self.emri['Nblocks'], self.slice_length)

        # Apply Tukey window once per block
        block_window_data = self.xp.array(tukey(self.slice_length, alpha=self.emri['alpha_block'], use_gpu=self.use_gpu))
        self.d_slices *= block_window_data[None, None, :] 

        # Calculate FFTs and masked PSDs
        freq = np.fft.rfftfreq(self.slice_length)/self.emri['dt']
        valid_freqs = freq[1:]

        self.freq_mask = np.full(len(valid_freqs), True)

        # fetch fmin and fmax with defaults to None
        fmin = self.emri.get("fmin", None)
        fmax = self.emri.get("fmax", None)

        #frequency lower bound
        if fmin is not None:
            self.freq_mask &= (valid_freqs >= fmin)
        if fmax is not None:
            self.freq_mask &= (valid_freqs <= fmax)

        freq_mask_xp = self.xp.array(self.freq_mask)

        PSD_coarse = generate_PSD(
            self.d_slices, 
            dt=self.emri['dt'], 
            noise_PSD=get_sensitivity,
            channels=self.channels, 
            noise_kwargs=self.noise_kwargs, 
            use_gpu=self.use_gpu
        )
        self.PSD_masked_xp = self.xp.atleast_2d(self.xp.array(PSD_coarse))[..., freq_mask_xp][:, None, :]
        self.d_fft_masked_xp = (self.emri['dt'] * self.xp.fft.rfft(self.d_slices, axis=-1))[..., 1:][..., freq_mask_xp]

    def build_priors(self):
        """
        Evaluates the physical trajectory and dynamically constructs the Eryn prior bounds.

        This method executes the true EMRI trajectory to find the exact injection values 
        at the start of every block. It then constructs Eryn `ProbDistContainer` objects 
        for the 'static' and 'evolving' parameter branches based strictly on the parameters 
        requested in the user's sample configuration JSON, ensuring periodic variables 
        (like phases) are properly registered.
        """

        print("Initializing priors...")
        
        kerr_traj = EMRIInspiral(func=KerrEccEqFlux)
        t, p, e, x, pp, pt, pr = kerr_traj(
            self.emri['m1'], self.emri['m2'], self.emri['a'], self.emri['p0'],
            self.emri['e0'], self.emri['xI0'], Phi_phi0=self.emri['Phi_phi0'],
            Phi_theta0=self.emri['Phi_theta0'], Phi_r0=self.emri['Phi_r0'],
            T=self.emri['T'], dt=self.emri['dt'], upsample=True
        )

        start_indices = np.arange(self.emri['Nblocks']) * self.slice_length
        self.true_evolving_dict = {
            "p0": p[start_indices],
            "e0": e[start_indices],
            "xI0": x[start_indices],
            "Phi_phi0": pp[start_indices] % (2*np.pi), #module important to respect prior bounds
            "Phi_theta0": pt[start_indices] % (2*np.pi),
            "Phi_r0": pr[start_indices] % (2*np.pi),
        }

        p_min, p_max = self.true_evolving_dict['p0'][-1] * (1-1e-2), self.emri['p0'] * (1+1e-2)
        e_min, e_max = (self.true_evolving_dict['e0'][-1] * (1-1e-2), self.emri['e0'] * (1+1e-2)) if self.emri['e0'] > 0 else (0.0, 0.01)
        a_spin = self.emri['a']
        if a_spin > 0:
            amin, amax = (a_spin * (1-1e-3), a_spin * (1+1e-3))
        elif a_spin < 0:
            amin, amax = (a_spin * (1+1e-3), a_spin * (1-1e-3))
        else:
            amin, amax = (-0.01, 0.01)

        self.bounds_dict = {
            "m1": (self.emri['m1'] * 0.999, self.emri['m1'] * 1.001),
            "m2": (self.emri['m2'] * 0.999, self.emri['m2'] * 1.001),
            "a": (amin, amax),
            "dist": (0.01, 10.0),
            "qS": (0.0, np.pi), 
            "phiS": (0.0, 2 * np.pi),
            "qK": (0.0, np.pi), 
            "phiK": (0.0, 2 * np.pi),
            "p0": (p_min, p_max), 
            "e0": (e_min, e_max), 
            "xI0": (0.0, 1.0),
            "Phi_phi0": (0.0, 2*np.pi), 
            "Phi_theta0": (0.0, 2*np.pi), 
            "Phi_r0": (0.0, 2*np.pi)
        }

        priors_evolving = {}
        priors_static = {}
        self.periodic = {"evolving":{}, "static":{}}

        for idx, param in enumerate(self.samp['evolving_params']):
            low, high = self.bounds_dict[param]
            priors_evolving[idx] = uniform_dist(low, high)
            if param in ["Phi_phi0", "Phi_theta0", "Phi_r0"]:
                self.periodic["evolving"][idx] = 2*np.pi

        for idx, param in enumerate(self.samp['static_params']):
            low, high = self.bounds_dict[param]
            priors_static[idx] = uniform_dist(low, high)
            if param in ['phiS','phiK']:
                self.periodic["static"][idx] = 2*np.pi 

        prob_dist_evolving = ProbDistContainer(priors_evolving)
        prob_dist_static = ProbDistContainer(priors_static)

        sigma_dict = self.emri.get(
            'sigma_prior', 
            {
            "p0": 1e-4,
            "e0": 1e-5,
            "xI0": 1e-5,
            "Phi_phi0": 0.5,
            "Phi_theta0": 0.5,
            "Phi_r0": 0.5
            }
        )

        dt_block_years = (self.slice_length * self.emri['dt']) / YRSID_SI

        self.priors = {
            "evolving": prob_dist_evolving,  # Required by HDFBackend
            "static": prob_dist_static,      # Required by HDFBackend
            "all_models_together": MarkovStudenttPrior(
                prior_ev=prob_dist_evolving,
                prior_st=prob_dist_static,
                dt_block=dt_block_years,
                nu=self.emri.get('nu_prior', 5.0),
                sigma_dict=sigma_dict,
                samp_config=self.samp,
                emri_config=self.emri,
                all_param_names=self.all_param_names,
                true_evolving_dict=self.true_evolving_dict,
                traj_instance=kerr_traj
            )
        }

    def run_sampler(self):
        """
        Configures and executes the Eryn MCMC sampler.

        This method acts as the engine of the SPLIT pipeline. It performs:
        1. File management, including resuming safely from old HDF5 backends.
        2. Dynamic index mapping for fixed vs. actively sampled parameters.
        3. Initializing and jittering the `start_state` array across walkers and temperatures.
        4. Setting up custom MCMC moves (Blocked Gibbs and Stretch).
        5. Booting the spawned multi-GPU Python `multiprocessing.Pool` for likelihood evaluations.
        6. Running the MCMC loop, checking Gelman-Rubin convergence, and saving diagnostic plots.
        """

        print("Initializing sampler...")

        # 1. Directory and File Management (Using your exact structure)
        nu = self.emri.get('nu_like', 5)
        Nblocks = self.emri['Nblocks']
        T = self.emri['T']
        nwalkers = self.samp['nwalkers']
        ntemps = self.samp['ntemps']

        # Set up folders
        folder = os.path.join(self.out_dir, f"nu_{nu}_Nblocks_{Nblocks}/")

        # Parse MCMC setup variables
        nsteps = self.samp.get('nsteps', 10000)
        check_interval = self.samp.get('check_interval', 100)
        burn = self.samp.get('burn', 0)
        thin_by = self.samp.get('thin_by', 1)
        
        # Parse filenames
        old_file_input = self.samp.get('old_filename', None)
        new_file_input = self.samp.get('filename', self.samp.get('new_filename', "default_run.h5"))
        
        # Construct the full file paths (incorporating your exact naming convention)
        new_filename = folder + f"T_{T}_nwalkers_{nwalkers}_ntemps_{ntemps}_{new_file_input}"
        old_filename = folder + f"T_{T}_nwalkers_{nwalkers}_ntemps_{ntemps}_{old_file_input}" if old_file_input else None

        #diagnostics folder always from new_filename
        diagnostics = os.path.join(folder, f"diagnostics_T_{T}_nwalkers_{nwalkers}_ntemps_{ntemps}_{new_file_input}/")
        os.makedirs(folder, exist_ok=True)
        os.makedirs(diagnostics, exist_ok=True)

        # 2. Map JSON string names to numeric indices for likelihood function
        indices_ev_in = [self.all_param_names.index(name) for name in self.samp['evolving_params']]
        indices_static_in = [self.all_param_names.index(name) for name in self.samp['static_params']]
        indices_ev_fixed = [self.all_param_names.index(name) for name in self.samp['fixed_evolving']]
        
        # Identify any parameters that were completely omitted from the JSON and fix them automatically
        active_and_fixed_ev = self.samp['evolving_params'] + self.samp['fixed_evolving']
        static_fixed_names = [p for p in self.all_param_names if p not in active_and_fixed_ev and p not in self.samp['static_params']]
        indices_static_fixed = [self.all_param_names.index(name) for name in static_fixed_names]
        
        value_fixed_ev = np.column_stack([self.true_evolving_dict[name] for name in self.samp['fixed_evolving']])
        value_fixed_static = [self.true_pars[idx] for idx in indices_static_fixed]

        ndim_evolving = len(indices_ev_in)
        ndim_static = len(indices_static_in)
        ndims = {"evolving": ndim_evolving, "static": ndim_static}
        nleaves_max = {"evolving": Nblocks, "static": 1}

        # 3. Initialize Backend and Start State
        # Extract true coordinates to act as the epicenter for the initial state
        val_samp_st = np.array([self.true_pars[idx] for idx in indices_static_in])
        val_samp_ev = np.column_stack([self.true_evolving_dict[name] for name in self.samp['evolving_params']])
        
        try:
            if old_filename is not None:
                # user provided an old filename, so we try to load it and resume from the last sample
                old_backend = HDFBackend(old_filename)
                start_state = old_backend.get_last_sample()
                # We will write new samples to the new_filename
                backend = HDFBackend(new_filename) 
                resume = True
            else:
                # no old filename provided. Check if new_filename already exists, and if so, append to it.
                backend = HDFBackend(new_filename)
                start_state = backend.get_last_sample()
                resume = True
            print(f"Resuming from saved state. Writing to: {new_filename}")

        except Exception as e:
            # start afresh if either of the above two conditions fail.
            resume = False
            print(f"Resume run failed with Exception {e}")
            print(f"New chain initiated. File not found or empty. Creating: {new_filename}")
            backend = HDFBackend(new_filename)
            
            jitter = 1e-4 #initial jitter for seeding walkers around true values

            # Create Static State
            coords_static = np.tile(val_samp_st, (ntemps, nwalkers, 1, 1))
            scale_st = np.abs(val_samp_st)
            scale_st[scale_st==0] = 1.0
            coords_static += np.random.normal(0, scale_st * jitter, size=coords_static.shape)

            # Force strict bounding so walkers do not start outside prior range
            for idx, param in enumerate(self.samp['static_params']):
                low, high = self.bounds_dict[param]
                coords_static[..., idx] = np.clip(coords_static[..., idx], low, high)

            # Create and Jitter Evolving State
            coords_evolving = np.tile(val_samp_ev, (ntemps, nwalkers, 1, 1))
            scale_ev = np.abs(val_samp_ev)
            scale_ev[scale_ev == 0] = 1.0
            coords_evolving += np.random.normal(0, scale_ev * jitter, size=coords_evolving.shape)

            for idx, param in enumerate(self.samp['evolving_params']):
                low, high = self.bounds_dict[param]
                coords_evolving[..., idx] = np.clip(coords_evolving[..., idx], low, high)

            start_state = State({"evolving": coords_evolving, "static": coords_static})

        # 4. Set up MCMC Architecture and Backend
        #cov = {
        #    "evolving": np.diag(np.ones(ndim_evolving)) * 1e-9,
        #    "static": np.diag(np.ones(ndim_static)) * 1e-9,
        #}
        
        # Blocked Gibbs sampling over individual leaves (Blocks). 
        # The covariance matrix for Gaussian kernel is adaptively modified. 
        # Our likelihood is independent across blocks (conditioned on the static parameters), so this is optimal.
        custom_gibbs_move = SequentialAdaptiveBlockedGibbsGaussianMove(reg=1e-9)

        # We also use a Blocked Stretch move which also respects the multi-branch structure to ensure decent acceptance rate.
        # update the hyperparameters with probability 1/(Nblocks+1). 
        # This way, the probability of updating the hyper parameters and specific block parameters is balanced.
        # Each block (leaf) gets updated with probability (1-prob_hyper)/Nblocks.
        custom_stretch_move = BlockedStretchMove(a=2.0, prob_hyper=1.0/(Nblocks+1))

        mixed_moves = [
            (StretchMove(), 0.2), #for global exploration. Acceptance rate expected to be poor.
            (custom_stretch_move, 0.2), 
            (custom_gibbs_move, 0.6)
        ]

        # 5. Initialize Multi-GPU pool
        num_processes = cp.cuda.runtime.getDeviceCount() * 2
        print(f"Booting Eryn MCMC with {num_processes} workers...")
        ctx = mp.get_context("spawn") ### SEQUENTIAL CUPY ARRAY INITIALIZATION FOR WORKERS (to avoid GPU memory conflicts)

        tdi_kwargs_base = dict(order=25, tdi="1st generation", tdi_chan="AE")
        T_block = self.emri['T']/Nblocks
        Tobs_block_padded = T_block + 2*(self.t_buffer/YRSID_SI)
        df = (self.slice_length * self.emri['dt']) ** -1

        with ctx.Pool(
            processes=num_processes,
            initializer=init_worker,
            initargs=(
                cp.asnumpy(self.d_fft_masked_xp), 
                cp.asnumpy(self.PSD_masked_xp), 
                cp.asnumpy(self.xp.array(self.freq_mask)), 
                Tobs_block_padded, self.emri['dt'], tdi_kwargs_base, 
                Nblocks, self.slice_length, self.index_lambda, self.index_beta, self.t_buffer
            )
        ) as pool:
            
            sampler = EnsembleSampler(
                nwalkers, ndims, log_like_semicoherent, self.priors, 
                nleaves_max=nleaves_max, nleaves_min=nleaves_max,
                kwargs=dict(
                    Tobs_block_padded=Tobs_block_padded, dt=self.emri['dt'], df=df, 
                    slice_length=self.slice_length, alpha_block=self.emri['alpha_block'],
                    len_pars_names=len(self.all_param_names), 
                    indices_ev_in=indices_ev_in, indices_static_in=indices_static_in, 
                    indices_ev_fixed=indices_ev_fixed, indices_static_fixed=indices_static_fixed, 
                    value_fixed_static=value_fixed_static, value_fixed_ev=value_fixed_ev, 
                    nu=self.emri['nu_like']
                ),
                moves=mixed_moves,
                tempering_kwargs=dict(ntemps=ntemps, adaptive=True, stop_adaptation=2000, permute=True),
                backend=backend, periodic=self.periodic, branch_names=["evolving", "static"],
                vectorize=False, pool=pool
            )

            traj_param_names = ["m1", "m2", "a", "p0", "e0", "xI0", "Phi_phi0", "Phi_theta0","Phi_r0"]
            traj_indices = [self.all_param_names.index(name) for name in traj_param_names]

            print("Starting MCMC loop...")

            current_state = start_state

            if burn > 0 and not resume:
                print(f"Running burn-in steps...")
                for burn_state in sampler.sample(current_state, iterations=burn, store=False, progress=True):
                    pass
                current_state = burn_state

            for sample in sampler.sample(current_state, iterations=nsteps, progress=True, thin_by=thin_by):
            
                if (sampler.iteration % check_interval == 0):

                    min_autocorr_iters = 5 #minimum number of iterations (N = min_autocorr_iters * tau) for reliable tau calcs. 

                    update_diagnostic_plots(
                        sampler, diagnostics, Nblocks, self.emri['dt'], self.slice_length, 
                        self.samp['evolving_params'], self.samp['static_params'], value_fixed_static, value_fixed_ev,
                        indices_static_in, indices_ev_in, indices_static_fixed, indices_ev_fixed,
                        self.all_param_names, self.true_pars, traj_indices, kerr_traj, 
                        val_samp_ev=val_samp_ev, val_samp_st=val_samp_st,
                        min_autocorr_iters=min_autocorr_iters
                    )

                    # get the lowest temperature chains by setting axis=1 index to 0
                    chain_st = sampler.get_chain()["static"][:, 0, :, 0, :] #chain_st has shape (nsteps, nwalkers, ndim)
                    chain_ev = sampler.get_chain()["evolving"][:, 0, :, :, :] #chain_ev has shape (nsteps, nwalkers, Nblocks, ndim)
                    
                    #quiet = True ensures that an AutocorrError is not thrown if Niter too small for tau estimate.
                    tau_st = emcee.autocorr.integrated_time(chain_st, tol=min_autocorr_iters, quiet=True)
                    # Calculate tau for each block independently!
                    tau_ev_blocks = []
                    for i in range(Nblocks):
                        tau_b = emcee.autocorr.integrated_time(chain_ev[:, :, i, :], tol=min_autocorr_iters, quiet=True)
                        tau_ev_blocks.append(tau_b)
                    tau_ev = np.array(tau_ev_blocks)

                    tau_est = max(np.nanmax(tau_st), np.nanmax(tau_ev))
                    
                    r_hat_st = compute_rhat(chain_st)
                    r_hat_ev = np.array([compute_rhat(chain_ev[:, :, i, :]) for i in range(Nblocks)])

                    #convergence criterion.
                    converged_tau = sampler.iteration > (50 * tau_est)
                    converged_r = np.all(np.nanmax(r_hat_st) < 1.05) and np.all(np.nanmax(r_hat_ev) < 1.05)

                    ######### PRINT STATEMENTS ##############
                    print("\n--- Sampler Status ---")
                    print(f"Iteration: {sampler.iteration}")
                    print(f"Max Gelman-Rubin (R-hat): Static = {np.nanmax(r_hat_st):.4f}, Evolving = {np.nanmax(r_hat_ev):.4f}")
                    print(f"Estimated Autocorrelation Time (tau): {tau_est:.1f} steps")

                    ess_per_walker = sampler.iteration / tau_est
                    print(f"Effective Sample Size per walker: ~{ess_per_walker:.1f}")

                    print("\n--- Acceptance Fractions by Move ---")
                    for i, (move, weight) in enumerate(zip(sampler.moves, sampler.weights)):
                        # move.acceptance_fraction is an array of shape (ntemps, nwalkers)
                        acc_frac_array = move.acceptance_fraction
                        mean_acc = np.mean(acc_frac_array)
                        print(f"Move {i} ({move.__class__.__name__}) | Weight: {weight:.2f} | Mean Acceptance: {mean_acc:.4f}")

                    # The backend stores the total number of accepted jumps per walker
                    total_accepted = sampler.backend.accepted
                    global_acc_frac = total_accepted / sampler.backend.iteration
                    print(f"Global Mean Acceptance Fraction: {np.mean(global_acc_frac):.4f}\n")
                    #########################################

                    if self.samp.get("check_converge", True) and (converged_tau and converged_r):
                        print(f"Convergence achieved at step {sampler.iteration}.")
                        break

            print("Run finished!")