import numpy as np
from typing import Optional
import os

from few.trajectory.ode import KerrEccEqFlux
from few.trajectory.inspiral import EMRIInspiral
from few.waveform.base import SphericalHarmonicWaveformBase
from few.utils.baseclasses import BackendLike
from few.utils.baseclasses import KerrEccentricEquatorial
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.summation.fdinterp import FDInterpolatedModeSum
from few.utils.modeselector import ModeSelector, NeuralModeSelector
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq

# get path to this file
from few.waveform.waveform import dir_path
dir_path = dir_path

class KerrEccEqAccFlux(KerrEccEqFlux):
    """
    KerrEccEq EMRI trajectory with a power-law modification to the p and e fluxes 
    induced by planetary migration effect in EccEq EMRIs due to an accretion disk.
    Model from Duque et al. (2024) https://arxiv.org/pdf/2411.03436.
    """

    def add_fixed_parameters(self, m1: float, m2: float, a:float, additional_args=None):
        """
        Additional trajectory parameters are initialized here. 
        For the accretion model, valid list of 
        """
        self.epsilon = m2/m1    # small mass ratio
        self.a = a              # primary spin

        if additional_args is None:
            is_vacuum = True
        elif len(additional_args) == 0:
            is_vacuum = True
        elif len(additional_args) == 1 and additional_args[0] in [None, 0.0]:
            is_vacuum = True
        else:
            is_vacuum = False

        if is_vacuum:
            self.num_add_args = 0
            self.Sigma0, self.h0, self.Sigma_p = (0.0, 0.0, 0.0)
            self.C_sub = 0.0
        else:
            try:
                self.Sigma0, self.h0, self.Sigma_p = additional_args
                self.num_add_args = len(additional_args)
                self.C_sub = 2.15 + 0.04 * self.Sigma_p
            except:
                raise ValueError(f"additional_args should be: [Sigma0, h0, Sigma_p]. Current input: {additional_args}")

    def modify_rhs(self, ydot: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        This function allows the user to modify the right-hand side of the ODE after any required Jacobian transforms
        have been applied.

        By default, this function returns the input right-hand side unchanged.
        """

        p, e = y[:2]                # orbital params
        pdot, edot = ydot[:2]       # their fluxes

        if self.Sigma0 == 0.0 and self.h0 == 0.0:
            edot_acc, pdot_acc = (0.0, 0.0)
        else:            
            # semi-major axis from p
            a_m = p/(1-e**2)

            # obtain disk properties at the current a_m
            r_10 = a_m / 10.0
            Sigma = self.Sigma0 * (r_10) ** (-self.Sigma_p)
            h = self.h0 * (r_10) ** ((2*self.Sigma_p-1)/4)
            
            Omega_K = a_m ** (-1.5)                 # dimensionless Keplerian frequency 

            # planetary migration timescale
            inv_t_gas = self.epsilon * (Sigma * a_m**2) * Omega_K/(h**4)

            # eccentricity evolution (Eq. 31)
            tgas_over_te = 0.78 * (1 - e**2)**(0.25) / (1 + (e/h)**3/30.0)
            inv_te = inv_t_gas * tgas_over_te
            edot_acc = -e * inv_te

            # semi-major axis evolution
            term1 = 1 - (e/(1.25*h))**4
            term2 = 1 + (e/(1.75*h))**5
            tgas_over_ta = 2 * self.C_sub * h**2 * (1-e**2) * (term1/term2)
            inv_ta = inv_t_gas * tgas_over_ta
            amdot_acc = -a_m * inv_ta

            # semi-latus rectum evolution from chain rule
            pdot_acc = amdot_acc * (1-e**2) - 2 * a_m * e * edot_acc

        # update flux array
        pdot_tot = pdot + pdot_acc
        edot_tot = edot + edot_acc

        ydot[:2] = (pdot_tot, edot_tot)

        return ydot

class FastKerrEccentricEquatorialAccretionFlux(
    SphericalHarmonicWaveformBase, KerrEccentricEquatorial
):
    """Prebuilt model for fast Kerr eccentric equatorial flux-based waveforms.

    This model combines the most efficient modules to produce the fastest
    accurate EMRI waveforms. It leverages GPU hardware for maximal acceleration,
    but is also available on for CPUs.

    The trajectory module used here is :class:`few.trajectory.inspiral` for a
    flux-based, sparse trajectory. This returns approximately 100 points.

    The amplitudes are then determined with
    :class:`few.amplitude.ampinterp2d.AmpInterp2D` along these sparse
    trajectories. This gives complex amplitudes for all modes in this model at
    each point in the trajectory. These are then filtered with
    :class:`few.utils.modeselector.ModeSelector`.

    The modes that make it through the filter are then summed by
    :class:`few.summation.interpolatedmodesum.InterpolatedModeSum`.

    See :class:`few.waveform.base.SphericalHarmonicWaveformBase` for information
    on inputs. See examples as well.

    args:
        inspiral_kwargs : Optional kwargs to pass to the
            inspiral generator. **Important Note**: These kwargs are passed
            online, not during instantiation like other kwargs here. Default is
            {}.
        amplitude_kwargs: Optional kwargs to pass to the
            amplitude generator during instantiation. Default is {}.
        sum_kwargs: Optional kwargs to pass to the
            sum module during instantiation. Default is {}.
        Ylm_kwargs: Optional kwargs to pass to the
            Ylm generator during instantiation. Default is {}.
        *args: args for waveform model.
        **kwargs: kwargs for waveform model.

    """

    def __init__(
        self,
        /,
        inspiral_kwargs: Optional[dict] = None,
        amplitude_kwargs: Optional[dict] = None,
        sum_kwargs: Optional[dict] = None,
        Ylm_kwargs: Optional[dict] = None,
        mode_selector_kwargs: Optional[dict] = None,
        force_backend: BackendLike = None,
        **kwargs: dict,
    ):
        if inspiral_kwargs is None:
            inspiral_kwargs = {}

        if "func" not in inspiral_kwargs.keys():
            inspiral_kwargs["func"] = KerrEccEqAccFlux      # <--- modified trajectory module

        # inspiral_kwargs = augment_ODE_func_name(inspiral_kwargs)

        if sum_kwargs is None:
            sum_kwargs = {}
        mode_summation_module = InterpolatedModeSum
        if "output_type" in sum_kwargs:
            if sum_kwargs["output_type"] == "fd":
                mode_summation_module = FDInterpolatedModeSum

        if mode_selector_kwargs is None:
            mode_selector_kwargs = {}
        mode_selection_module = ModeSelector
        if "mode_selection_type" in mode_selector_kwargs:
            if mode_selector_kwargs["mode_selection_type"] == "neural":
                mode_selection_module = NeuralModeSelector
                if "mode_selector_location" not in mode_selector_kwargs:
                    mode_selector_kwargs["mode_selector_location"] = os.path.join(
                        dir_path,
                        "./files/modeselector_files/KerrEccentricEquatorialFlux/",
                    )
                mode_selector_kwargs["keep_inds"] = np.array(
                    [0, 1, 2, 3, 4, 6, 7, 8, 9]
                )

        KerrEccentricEquatorial.__init__(
            self,
            **{
                key: value
                for key, value in kwargs.items()
                if key in ["lmax", "nmax", "ndim"]
            },
            force_backend=force_backend,
        )
        SphericalHarmonicWaveformBase.__init__(
            self,
            inspiral_module=EMRIInspiral,
            amplitude_module=AmpInterpKerrEccEq,
            sum_module=mode_summation_module,
            mode_selector_module=mode_selection_module,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            mode_selector_kwargs=mode_selector_kwargs,
            **{
                key: value for key, value in kwargs.items() if key in ["normalize_amps"]
            },
            force_backend=force_backend,
        )

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    @property
    def allow_batching(self):
        return False

    def __call__(
        self,
        m1: float,
        m2: float,
        a: float,
        p0: float,
        e0: float,
        xI: float,
        theta: float,
        phi: float,
        *args: Optional[tuple],
        **kwargs: Optional[dict],
    ) -> np.ndarray:
        """
        Generate the waveform.

        Args:
            m1: Mass of larger black hole in solar masses.
            m2: Mass of compact object in solar masses.
            a: Dimensionless spin of massive black hole.
            p0: Initial semilatus rectum of inspiral trajectory.
            e0: Initial eccentricity of inspiral trajectory.
            xI: Initial cosine of the inclination angle.
            theta: Polar angle of observer.
            phi: Azimuthal angle of observer.
            *args: Placeholder for additional arguments.
            **kwargs: Placeholder for additional keyword arguments.

        Returns:
            Complex array containing generated waveform.

        """
        return self._generate_waveform(
            m1,
            m2,
            a,
            p0,
            e0,
            xI,
            theta,
            phi,
            *args,
            **kwargs,
        )