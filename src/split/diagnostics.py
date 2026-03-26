import os
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from few.utils.constants import YRSID_SI
from tqdm import tqdm

def _plot_static_diagnostics(chain, names, truths, discard_idx, out_dir):
    """Plots 1D walks and corner plots for the static branch."""
    nsteps, nwalkers, ndim = chain.shape
    
    # 1D Walks
    fig_1d, axs = plt.subplots(ndim, 1, figsize=(10, 3 * ndim))
    # Ensure axs is iterable even if ndim == 1
    if ndim == 1: axs = [axs] 
    
    for j in range(ndim):
        axs[j].plot(chain[:, :, j], alpha=0.5)
        axs[j].axhline(truths[j], color='k', linestyle='--', label='True value')
        axs[j].set_ylabel(names[j])
        if j == 0: axs[j].legend()
    axs[-1].set_xlabel("Steps")
    fig_1d.tight_layout()
    plt.savefig(os.path.join(out_dir, "1dplots_static.png"), dpi=300)
    plt.close(fig_1d)

    # Corner Plot
    flat_chain = chain[discard_idx:].reshape(-1, ndim)
    try:
        fig_corner = corner.corner(
            flat_chain, labels=names, truths=truths, 
            show_titles=True, quantiles=[0.16, 0.5, 0.84]
        )
        plt.savefig(os.path.join(out_dir, "corner_static.png"), dpi=300)
        plt.close(fig_corner)
    except ValueError as e:
        print(f"Skipping static corner plot: {e}")

def _plot_evolving_diagnostics(chain, names, truths, discard_idx, Nblocks, out_dir):
    """Plots 1D walks and corner plots for each block in the evolving branch."""
    nsteps, nwalkers, _, ndim = chain.shape

    for i in range(Nblocks):
        labels_i = [f"{name}_{i}" for name in names]
        
        # 1D Walks
        fig_1d, axs = plt.subplots(ndim, 1, figsize=(10, 3 * ndim))
        if ndim == 1: axs = [axs]

        for j in range(ndim):
            axs[j].plot(chain[:, :, i, j], alpha=0.5)
            axs[j].axhline(truths[i, j], color='k', linestyle='--', label='True value')
            axs[j].set_ylabel(labels_i[j])
            if j == 0: axs[j].legend()
        
        axs[-1].set_xlabel("Steps")
        fig_1d.tight_layout()
        plt.savefig(os.path.join(out_dir, f"1dplots_block_{i}.png"), dpi=300)
        plt.close(fig_1d)

        # Corner Plot
        flat_chain = chain[discard_idx:, :, i, :].reshape(-1, ndim)
        try:
            fig_corner = corner.corner(
                flat_chain, labels=labels_i, truths=truths[i], 
                show_titles=True, quantiles=[0.16, 0.5, 0.84]
            )
            plt.savefig(os.path.join(out_dir, f"corner_evolving_{i}.png"), dpi=300)
            plt.close(fig_corner)
        except ValueError as e:
            print(f"Skipping evolving corner plot for block {i}: {e}")

def _plot_autocorrelation(chain_st, chain_ev, out_dir, min_iters):
    """Plots the integrated autocorrelation time for both branches."""
    nsteps, nwalkers, ndim_st = chain_st.shape
    _, _, Nblocks, ndim_ev = chain_ev.shape

    if nsteps <= 100:
        return

    N_steps = np.exp(np.linspace(np.log(100), np.log(nsteps), 50)).astype(int)
    taus_st = np.empty((len(N_steps), ndim_st))
    taus_ev = np.empty((len(N_steps), ndim_ev))

    for idx, n in enumerate(N_steps):
        taus_st[idx] = emcee.autocorr.integrated_time(chain_st[:n], tol=min_iters, quiet=True)
        
        reshaped_ev = chain_ev[:n].transpose(0, 1, 2, 3).reshape(n, nwalkers * Nblocks, ndim_ev)
        taus_ev[idx] = emcee.autocorr.integrated_time(reshaped_ev, tol=min_iters, quiet=True)

    fig_ac, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    for j in range(ndim_st): ax1.loglog(N_steps, taus_st[:, j], "b-", alpha=0.3)
    ax1.loglog(N_steps, N_steps / 50.0, "--r", label=r"$\tau = N/50$")
    ax1.set_title("Autocorrelations: Static Branch")
    ax1.legend()

    for j in range(ndim_ev): ax2.loglog(N_steps, taus_ev[:, j], "g-", alpha=0.3)
    ax2.loglog(N_steps, N_steps / 50.0, "--r")
    ax2.set_title("Autocorrelations: Evolving Branch (Max across blocks)")
    
    fig_ac.tight_layout()
    plt.savefig(os.path.join(out_dir, "autocorr.png"), dpi=300)
    plt.close(fig_ac)

def _plot_backward_projection(chain_st, chain_ev, discard_idx, names_st, names_ev, 
                              true_pars_all, Nblocks, traj_config, out_dir):
    """Evolves parameters backwards to t=0 and plots the joint posterior."""
    nsteps = chain_st.shape[0]
    if nsteps <= 100:
        return

    print("Evolving samples backwards to t=0 for joint posterior...")
    
    recent_st = chain_st[discard_idx:].reshape(-1, chain_st.shape[-1])
    recent_ev = chain_ev[discard_idx:].reshape(-1, Nblocks, chain_ev.shape[-1])
    n_samples = len(recent_st)

    # Unpack config
    dt, slice_len = traj_config["dt"], traj_config["slice_length"]
    idx_st_in, idx_ev_in = traj_config["idx_st_in"], traj_config["idx_ev_in"]
    idx_st_fix, idx_ev_fix = traj_config["idx_st_fix"], traj_config["idx_ev_fix"]
    val_st_fix, val_ev_fix = traj_config["val_st_fix"], traj_config["val_ev_fix"]
    kerr_traj = traj_config["kerr_traj_instance"]
    traj_idx = traj_config["traj_indices"]
    total_pars_len = traj_config["total_pars_len"]

    projected_t0_samples = []

    for j in tqdm(range(n_samples),desc='backwards projection'):
        for i in range(Nblocks):
            T_backwards = (i * slice_len * dt) / YRSID_SI

            if T_backwards == 0.0:
                projected_t0_samples.append(list(recent_st[j]) + list(recent_ev[j, i, :]))
                continue

            pars_block = np.zeros(total_pars_len)
            pars_block[idx_st_in] = recent_st[j]
            pars_block[idx_ev_in] = recent_ev[j, i, :]
            pars_block[idx_st_fix] = val_st_fix
            pars_block[idx_ev_fix] = val_ev_fix[i, :]

            args = pars_block[traj_idx]

            try:
                traj_output = kerr_traj(
                    *args[:-3], 
                    Phi_phi0=-args[-3], Phi_theta0=-args[-2], Phi_r0=-args[-1], 
                    T=T_backwards, dt=dt, upsample=False, integrate_backwards=True
                )
                
                proj_ev_dict = {
                    "p0": traj_output[1][-1],
                    "e0": traj_output[2][-1],
                    "xI0": traj_output[3][-1],
                    "Phi_phi0": -traj_output[4][0] % (2*np.pi),
                    "Phi_theta0": -traj_output[5][0] % (2*np.pi),
                    "Phi_r0": -traj_output[6][0] % (2*np.pi)
                }
                
                # Dynamically extract only the actively sampled parameters in the correct order
                proj_ev = [proj_ev_dict[name] for name in names_ev]
                
                projected_t0_samples.append(list(recent_st[j]) + proj_ev)
            except Exception:
                continue

    if projected_t0_samples:
        projected_t0_samples = np.array(projected_t0_samples)
        
        active_truths = [true_pars_all[idx] for idx in idx_st_in] + [true_pars_all[idx] for idx in idx_ev_in]
        
        try:
            fig_t0 = corner.corner(
                projected_t0_samples, labels=(names_st + names_ev), truths=active_truths, 
                show_titles=True, quantiles=[0.16, 0.5, 0.84]
            )
            plt.savefig(os.path.join(out_dir, "corner_t0_projected.png"), dpi=300)
            plt.close(fig_t0)
        except ValueError as e:
            print(f"Skipping projected corner plot: {e}")

# ==========================================
# MAIN WRAPPER FUNCTION
# ==========================================
def update_diagnostic_plots(sampler, diagnostics_dir, Nblocks, 
                            static_in_names, ev_in_names,
                            val_samp_st, val_samp_ev, true_pars_all, 
                            traj_config, min_autocorr_iters=50, discard_frac=0.5):
    """
    Extract multi-branch chains, plot 1D walks, static posteriors, and t=0 projections.
    
    traj_config (dict): Contains all trajectory mapping parameters:
        dt, slice_length, idx_st_in, idx_ev_in, idx_st_fix, idx_ev_fix, 
        val_st_fix, val_ev_fix, kerr_traj_instance, traj_indices, total_pars_len.
    """
    print(f"\n[Step {sampler.iteration}] Generating diagnostic plots...")
    os.makedirs(diagnostics_dir, exist_ok=True)

    # Extract chains. Shape mappings:
    # chain_static: (nsteps, nwalkers, ndim_static)
    # chain_evolving: (nsteps, nwalkers, Nblocks, ndim_evolving)
    chain_st = sampler.get_chain()["static"][:, 0, :, 0, :] 
    chain_ev = sampler.get_chain()["evolving"][:, 0, :, :, :] 

    # Discard the first 50% for corner plots and backwards evolution
    discard_idx = int(chain_st.shape[0] * discard_frac)

    # 1. Static Branch Diagnostics
    _plot_static_diagnostics(chain_st, static_in_names, val_samp_st, discard_idx, diagnostics_dir)

    # 2. Evolving Branch Diagnostics
    _plot_evolving_diagnostics(chain_ev, ev_in_names, val_samp_ev, discard_idx, Nblocks, diagnostics_dir)

    # 3. Autocorrelation
    _plot_autocorrelation(chain_st, chain_ev, diagnostics_dir, min_autocorr_iters)

    # 4. Backward Projection to t=0
    _plot_backward_projection(
        chain_st, chain_ev, discard_idx, static_in_names, ev_in_names, 
        true_pars_all, Nblocks, traj_config, diagnostics_dir
    )