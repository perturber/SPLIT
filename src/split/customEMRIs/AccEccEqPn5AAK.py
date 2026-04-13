import numpy as np
from typing import Optional

from few.trajectory.ode.pn5 import PN5
from few.trajectory.inspiral import EMRIInspiral
from few.summation.aakwave import AAKSummation
from few.waveform.base import AAKWaveformBase
from few.utils.baseclasses import BackendLike

class AccEccEqPN5Traj(PN5):
    """
    PN5 EMRI trajectory with a power-law modification to the p and e fluxes 
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
    
class AccEccEqPn5AAKWaveform(AAKWaveformBase):
    r"""Waveform generation class for AAK with AccEccEqPN5Traj trajectory as defined above.

    This class generates waveforms based on the Augmented Analytic Kludge
    given in the
    `EMRI Kludge Suite <https://github.com/alvincjk/EMRI_Kludge_Suite/>`_.
    However, here the trajectory is vastly improved by employing the 5PN
    fluxes for generic Kerr orbits from
    `Fujita & Shibata 2020 <https://arxiv.org/abs/2008.13554>`_.

    The 5PN trajectory produces orbital and phase trajectories.
    The trajectory is calculated until the orbit reaches
    within 0.2 of the separatrix, determined from
    `arXiv:1912.07609 <https://arxiv.org/abs/1912.07609/>`_. The
    fundamental frequencies along the trajectory at each point are then
    calculated from the orbital parameters and the spin value given by (`Schmidt 2002 <https://arxiv.org/abs/gr-qc/0202090>`_).

    These frequencies along the trajectory are then used to map to the
    frequency basis of the `Analytic Kludge <https://arxiv.org/abs/gr-qc/0310125>`_. This mapping
    takes the form of time evolving large mass and spin parameters, as
    well as the use of phases and frequencies in
    :math:`(alpha, \Phi, \gamma)`:

    .. math:: \Phi = \Phi_\phi,

    .. math:: \gamma = \Phi_\phi + \Phi_\Theta,

    .. math:: alpha = \Phi_\phi + \Phi_\Theta + \Phi_r.

    The frequencies in that basis are found by taking the time derivatives
    of each equation above.

    This class has GPU capabilities and works from the sparse trajectory
    methodoligy with cubic spine interpolation of the smoothly varying
    waveform quantities. This waveform does not have the freedom in terms
    of user-chosen quantitites that
    :class:`few.waveform.base.SphericalHarmonicWaveformBase` contains.
    This is mainly due to the specific waveform constructions particular
    to the AAK/AK.

    **Please note:** the 5PN trajectory and AAK waveform take the parameter
    :math:`Y\equiv\cos{\iota}=L/\sqrt{L^2 + Q}` rather than :math:`x_I` as is accepted
    for relativistic waveforms and in the generic waveform interface discussed above.
    The generic waveform interface directly converts :math:`x_I` to :math:`Y`.

    args:
        inspiral_kwargs: Optional kwargs to pass to the
            inspiral generator. **Important Note**: These kwargs are passed
            online, not during instantiation like other kwargs here. Default is
            {}. This is stored as an attribute.
        sum_kwargs: Optional kwargs to pass to the
            sum module during instantiation. Default is {}.
    """

    def __init__(
        self,
        inspiral_kwargs: Optional[dict] = None,
        sum_kwargs: Optional[dict] = None,
        force_backend: BackendLike = None,
    ):
        if inspiral_kwargs is None:
            inspiral_kwargs = {}
        if "func" not in inspiral_kwargs.keys():
            inspiral_kwargs["func"] = AccEccEqPN5Traj   # <-- modified trajecotry module.

        AAKWaveformBase.__init__(
            self,
            inspiral_module=EMRIInspiral,
            sum_module=AAKSummation,
            inspiral_kwargs=inspiral_kwargs,
            sum_kwargs=sum_kwargs,
            force_backend=force_backend,
        )

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()
