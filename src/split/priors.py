import numpy as np

class MarkovStudenttPrior:
    """
    Custom Eryn Prior evaluating a heavy-tailed Student-t transition penalty 
    for semi-coherent EMRI blocks against theoretical vacuum-GR trajectories.
    
    This class bypasses the assumption that semi-coherent blocks are fully 
    independent. It applies a base prior (uniform) to 
    all blocks, and then evaluates a Markovian transition penalty: it forward-evolves 
    the parameters from block (i-1) using pure GR, and applies a Student-t penalty 
    based on how far the proposed parameters in block (i) deviate from that prediction.
    """
    def __init__(self, prior_ev, prior_st, dt_block, nu,
                 sigma_dict, samp_config, emri_config,
                 all_param_names, true_evolving_dict, 
                 traj_instance, traj_add_args):
        """
        Initializes the Student-t Prior.

        Parameters
        ----------
        prior_ev : eryn.prior.ProbDistContainer
            The base prior bounds for the evolving parameters. Used to enforce 
            hard cutoffs or tapered edges on the parameter space.
        prior_st : eryn.prior.ProbDistContainer
            The base prior bounds for the static parameters.
        dt_block : float
            The physical duration of a single semi-coherent block, in years. 
            Used as the time step to forward-evolve the trajectory.
        nu : float
            Degrees of freedom for the heavy-tailed Student-t distribution. 
            Lower values create heavier tails, allowing larger deviations from GR.
        sigma_dict : dict
            Dictionary mapping evolving parameter names to their allowed standard 
            deviation (tolerance). Example: {"p0": 1e-4, "Phi_phi0": 0.5}.
        samp_config : dict
            The sampling configuration dictionary loaded from JSON. Must contain keys:
            'evolving_params', 'static_params', and 'fixed_evolving'.
        emri_config : dict
            The physical EMRI configuration dictionary loaded from JSON. Used as a 
            fallback to extract static macro-parameters (m1, m2, a) if they are fixed.
        all_param_names : list of str
            A complete list of all 14 standard Kerr parameter names used in the pipeline.
        true_evolving_dict : dict
            Dictionary containing arrays of the true injected parameter values for 
            each block. Used to look up the exact values of parameters defined in 'fixed_evolving'.
        traj_instance : few.trajectory.inspiral.EMRIInspiral
            A pre-initialized FEW trajectory object (e.g., initialized with KerrEccEqFlux). 
            Used to compute the deterministic forward evolution between blocks.
        traj_add_args : List
            A list of additional arguments in case of a custom trajectory.
        """
        self.prior_ev = prior_ev
        self.prior_st = prior_st
        self.dt_block = dt_block
        self.nu = nu

        self.samp = samp_config
        self.emri = emri_config
        self.all_names = all_param_names

        self.ev_names = self.samp['evolving_params']
        self.st_names = self.samp['static_params']
        self.fix_ev = self.samp['fixed_evolving']
        self.true_ev_dict = true_evolving_dict

        self.D = len(self.ev_names)
        self.inv_var = np.zeros(self.D)
        for i, name in enumerate(self.ev_names):
            if name in sigma_dict:
                self.inv_var[i] = 1.0 / (sigma_dict[name]**2)
            else:
                self.inv_var[i] = 1.0 / (100.0**2) #fall back
        
        self.traj = traj_instance
        self.traj_add_args = traj_add_args

        # Satisfy the Eryn gods
        self.key_order = [0]

    def logpdf(self, coords, inds, supps=None, branch_supps=None):
        """
        Evaluates the joint log-prior probability for the entire MCMC ensemble.

        This method is called natively by Eryn when "all_models_together" is specified.
        It first evaluates the independent base bounds. If a walker is within bounds, 
        it computes the autoregressive transition penalty block-by-block.

        Parameters
        ----------
        coords : dict
            Dictionary containing the proposed coordinates for all walkers.
            - coords["evolving"] shape: (ntemps, nwalkers, Nblocks, ndim_ev)
            - coords["static"] shape: (ntemps, nwalkers, 1, ndim_st)
        inds : dict
            Dictionary of boolean arrays indicating which leaves/blocks are active.
            - inds["evolving"] shape: (ntemps, nwalkers, Nblocks)
        supps : object, optional
            Supplemental overall information passed by Eryn. (Unused here)
        branch_supps : dict, optional
            Supplemental branch-specific information passed by Eryn. (Unused here)

        Returns
        -------
        total_logP + penalty : np.ndarray
            A 2D array of shape (ntemps, nwalkers) containing the final computed 
            log-prior probability (base bounds + trajectory penalty) for each walker.
            Walkers proposing unphysical jumps or values outside the base bounds 
            will return -inf.
        """
        evolving = coords["evolving"]
        static = coords["static"]

        ntemps, nwalkers, Nblocks, ndim_ev = evolving.shape
        _, _, nleaves_st, ndim_st = static.shape

        # Eryn's ProbDistContainer requires 2D arrays. 
        # Flatten the first 3 dimensions into a single batch dimension.
        ev_flat = evolving.reshape(-1, ndim_ev)
        st_flat = static.reshape(-1, ndim_st)

        logP_ev_flat = self.prior_ev.logpdf(ev_flat)
        logP_st_flat = self.prior_st.logpdf(st_flat)

        # Reshape the output log-probabilities back to their native shapes
        logP_ev = logP_ev_flat.reshape(ntemps, nwalkers, Nblocks)
        logP_st = logP_st_flat.reshape(ntemps, nwalkers, nleaves_st)

        logP_ev[~inds["evolving"]] = 0.0
        logP_st[~inds["static"]] = 0.0

        total_logP = np.sum(logP_ev, axis=-1) + np.sum(logP_st, axis=-1)

        # if outside base bounds, return -inf
        valid_mask = ~np.isinf(total_logP)
        if not np.any(valid_mask):
            return total_logP
        
        valid_indices = np.argwhere(valid_mask)
        penalty = np.zeros_like(total_logP)

        # Markovian Student-t Transition Probability
        for temp_idx, walker_idx in valid_indices:
            w_ev = evolving[temp_idx, walker_idx]
            w_st = static[temp_idx, walker_idx, 0]

            #Extract static hyper-parameters for trajectory
            m1 = w_st[self.st_names.index("m1")] if "m1" in self.st_names else self.emri["m1"]
            m2 = w_st[self.st_names.index("m2")] if "m2" in self.st_names else self.emri["m2"]
            a = w_st[self.st_names.index("a")] if "a" in self.st_names else self.emri["a"]

            walker_penalty = 0.0

            for i in range(1, Nblocks):
                state_prev = {}
                for param in ["p0","e0","xI0","Phi_phi0","Phi_theta0","Phi_r0"]:
                    if param in self.ev_names:
                        state_prev[param] = w_ev[i-1, self.ev_names.index(param)]
                    else:
                        state_prev[param] = self.emri[param]

                #forward evolve the block i-1 by dt_block
                try:
                    _, p, e, x, pp, pt, pr = self.traj(
                        m1, m2, a,
                        state_prev["p0"], state_prev["e0"], state_prev["xI0"],
                        *self.traj_add_args,
                        Phi_phi0=state_prev["Phi_phi0"], 
                        Phi_theta0=state_prev["Phi_theta0"], 
                        Phi_r0=state_prev["Phi_r0"],
                        T=self.dt_block, #no upsampling to save time
                    )
                    expected = {
                        "p0": p[-1], "e0": e[-1], "xI0": x[-1],
                        "Phi_phi0": pp[-1]%(2*np.pi), 
                        "Phi_theta0": pt[-1]%(2*np.pi), 
                        "Phi_r0":pr[-1]%(2*np.pi)
                    }

                except Exception:
                    walker_penalty += -np.inf
                    break

                # Compute penalty
                D2 = 0.0
                for j, param in enumerate(self.ev_names):
                    delta = w_ev[i, j] - expected[param]

                    if param in ["Phi_phi0", "Phi_theta0", "Phi_r0"]:
                        # make sure that the distance is calculated 
                        # correctly for phases by wrapping about the period.
                        delta = (delta + np.pi) % (2*np.pi) - np.pi

                    D2 += (delta**2) * self.inv_var[j]

                walker_penalty += -0.5 * (self.nu + self.D) * np.log(1.0 + D2 / self.nu)

            penalty[temp_idx, walker_idx] = walker_penalty

        return total_logP + penalty