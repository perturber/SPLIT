from eryn.moves import MHMove, RedBlueMove
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="cupy")

class SharedState:
    def __init__(self):
        self.step = 0

#define a global custom move class for probabilistic blocked Gibbs updates of the evolving parameters.
class BlockedGibbsGaussianMove(MHMove):
    """Probabilistic Blocked Gibbs sampler with fixed Gaussian proposals.
    On each step either the static (hyper) branch or one randomly chosen
    evolving leaf is updated with a multivariate Gaussian kernel.  The branch
    to update is selected with probability ``prob_hyper`` for the static branch
    and ``1 - prob_hyper`` for the evolving branch.  Pre-supplied covariance
    matrices are scaled by the parallel-tempering inverse temperature to
    preserve detailed balance across temperature levels.
    """

    def __init__(self, cov, prob_hyper=0.5,**kwargs):
        """
        Custom MHmove class for Blocked updates of the evolving parameters (leaves) and the hyper parameters (static branch).
        cov (dict): keys: branch names values: covariance matrices for those branches of shape (ndim_branch, ndim_branch).
        prob_hyper (float): probability that the hyper parameters will be updated (and not one of the source leaves)
        """
        self.cov_evolving = cov["evolving"]
        self.cov_static = cov["static"]
        self.prob_hyper = prob_hyper
        super().__init__(**kwargs)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """
        Generate the proposed state.
        branches_coords: dict with keys as branch names.
                         Values are (ntemps, nwalkers, nleaves_max, ndim)
        """
        q = {}

        s_stat = self.xp.asarray(branches_coords["static"])
        s_evol = self.xp.asarray(branches_coords["evolving"])

        ntemps, nwalkers, nleaves, ndim_evolving = s_evol.shape
        _, _, _, ndim_stat = s_stat.shape

        cov_stat_xp = self.xp.asarray(self.cov_static)
        cov_evol_xp = self.xp.asarray(self.cov_evolving)

        q = {"static": s_stat.copy(), "evolving": s_evol.copy()}

        rng = random if not getattr(self, "use_gpu", False) else self.xp.random

        # Extract inverse temperatures (betas) for tempering
        # Fallback to an array of 1.0s if tempering is not initialized
        betas = self.xp.ones(ntemps)
        if hasattr(self, "temperature_control") and self.temperature_control is not None:
            betas = self.xp.asarray(self.temperature_control.betas)

        #Randomly choose ONE branch to update
        if random.uniform() < self.prob_hyper:
            #update the hyper parameters (static branch).
            # generate steps for the static branch shape: (ntemps, nwalkers, 1 leaf, ndim_static)
            mean_stat = self.xp.zeros(len(self.cov_static))
            # Loop over temperatures to apply the scaled covariance matrix
            for t in range(ntemps):
                # Scale covariance by T = 1 / beta
                cov_t = cov_stat_xp / betas[t]
                
                static_step = rng.multivariate_normal(
                    mean_stat, cov_t, size=(nwalkers, 1)
                )
                q["static"][t] += static_step

        else:
            #update ONE of the leaves (blocks) of the evolving branch.
            # randomly choose a leaf (block) to update. This probability can also be weighted if needed. 
            leaf_idx = random.choice(nleaves)

            # generate step for the evolving branch for the chosen leaf of shape (ntemps, nwalkers, ndim_evolving)
            mean_evol = self.xp.zeros(len(self.cov_evolving))
            # Loop over temperatures to apply the scaled covariance matrix
            for t in range(ntemps):
                # Scale covariance by T = 1 / beta
                cov_t = cov_evol_xp / betas[t]
                
                evolving_step = rng.multivariate_normal(
                    mean_evol, cov_t, size=nwalkers
                )
                q["evolving"][t, :, leaf_idx, :] += evolving_step

        # symmetric proposal. log proposal ratio factor is 0
        factors = self.xp.zeros((ntemps,nwalkers))

        if getattr(self, "use_gpu", False) and not getattr(self, "return_gpu", False):
            q["static"] = q["static"].get()
            q["evolving"] = q["evolving"].get()
            factors = factors.get()

        return q, factors

# similarly, define a probabilistic blocked stretch move class 
class BlockedGibbsStretchMove(RedBlueMove):
    """Probabilistic Blocked Gibbs sampler using affine-invariant stretch proposals.
    Adapts the Goodman–Weare stretch move (RedBlue ensemble) to the two-branch
    SPLIT structure.  On each step either the static branch or one randomly chosen
    evolving leaf is updated via a stretch proposal along the line connecting an
    active and a complementary walker.  The correct log-factor
    ``(ndim - 1) * log(z)`` is accumulated for detailed balance.
    """

    def __init__(self, a=2.0, prob_hyper=0.5, **kwargs):
        """
        Custom RedBlue move class for Blocked updates of the evolving parameters (leaves) and the hyper parameters (static branch).
        StretchMove based on Goodman & Weare's affine-invariant move. Here, we adapt it for the multi-branch structure and blocked updates.
        a (float): stretch move scale parameter
        prob_hyper (float): probability that the hyper parameters will be updated (and not one of the source leaves) 
        """
        self.a = a
        self.prob_hyper = prob_hyper
        super().__init__(**kwargs)
    
    def get_proposal(self, s_all, c_all, random, **kwargs):
        """
        s_all (dict): Keys are ``branch_names`` and values are coordinates
            for which a proposal is to be generated.
        c_all (dict): Keys are ``branch_names`` and values are lists. These
            lists contain all the complement array values.

        Notes:
            self.xp comes from the parent RedBlueMove class.    
        """

        #extract and concatenate the static branch coordinates
        s_stat = self.xp.asarray(s_all["static"])
        c_stat = self.xp.concatenate([self.xp.asarray(c) for c in c_all["static"]], axis=1)

        #evolving branch
        s_evol = self.xp.asarray(s_all["evolving"])
        c_evol = self.xp.concatenate([self.xp.asarray(c) for c in c_all["evolving"]], axis=1)

        ntemps, Ns = s_stat.shape[0], s_stat.shape[1]
        Nc = c_stat.shape[1]

        #Draw random complementary walkers
        rint = random.randint(Nc, size=(ntemps, Ns))

        #Draw the stretch scale variable z from the standard distribution
        zz = ((self.a - 1.0) * random.rand(ntemps, Ns)+1.0) ** 2.0 / self.a

        #cast zz to Cupy array natively using self.xp
        zz = self.xp.asarray(zz)

        #the symmetry factor for acceptance ratio
        factors = self.xp.zeros((ntemps, Ns))

        # copy the current state so untouched blocks remain identical
        q = {"static": s_stat.copy(), "evolving": s_evol.copy()}

        if random.uniform() < self.prob_hyper:
            #update the static branch with stretch move
            ndim = s_stat.shape[-1]
            for t in range(ntemps):
                for w in range(Ns):
                    #get the complementary walker static parameters for this walker
                    c_val = c_stat[t, rint[t, w], 0, :]
                    #get the active walker static parameters for this walker
                    s_val = s_stat[t, w, 0, :]
                    #propose new static parameters by stretching along the line connecting s_val and c_val
                    q["static"][t, w, 0, :] = c_val - zz[t,w] * (c_val - s_val)

            #the factor scales by the dimensionality of the updated block
            #this is important for detailed balance of the stretch move.
            factors += (ndim - 1.0) * self.xp.log(zz)

        else:
            #update ONE of the evolving blocks with stretch move
            nleaves = s_evol.shape[2]
            ndim = s_evol.shape[-1]
            leaf_idx = random.choice(nleaves) #uniform choice

            for t in range(ntemps):
                for w in range(Ns): 
                    c_val = c_evol[t, rint[t, w], leaf_idx, :]
                    s_val = s_evol[t, w, leaf_idx, :]
                    q["evolving"][t, w, leaf_idx, :] = c_val - zz[t,w] * (c_val - s_val)

            factors += (ndim - 1.0) * self.xp.log(zz)

        if getattr(self, "use_gpu", False) and not getattr(self, "return_gpu", False):
            q["static"] = q["static"].get()
            q["evolving"] = q["evolving"].get()
            factors = factors.get()

        return q, factors
    
class SequentialBlockedGibbsGaussianMove(MHMove):
    """Deterministic sequential Blocked Gibbs sampler with fixed Gaussian proposals.
    Cycles through all evolving blocks and the static branch, updating exactly 
    one block per call.  A ``SharedState`` object keeps the schedule synchronized 
    when this move is mixed with other sequential moves 
    (e.g. ``SequentialBlockedGibbsStretchMove``).  The fixed covariance
    matrices are pre-scaled by the inverse temperature at each step.
    """

    def __init__(self, cov, shared_state=None, **kwargs):
        """
        Custom MHmove class for sequential updates of the
        evolving parameters (leaves) and hyper parameters (static branch).
        cov (dict): keys: branch names values: covariance matrices for those branches.
        shared_state (SharedState): Object to synchronize the iteration step across mixed moves.
        """
        self.cov_evolving = cov["evolving"]
        self.cov_static = cov["static"]

        # Internal step counter to track the Gibbs sweep
        self.shared_state = shared_state if shared_state is not None else SharedState()

        super().__init__(**kwargs)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        q = {}

        s_stat = self.xp.asarray(branches_coords["static"])
        s_evol = self.xp.asarray(branches_coords["evolving"])

        ntemps, nwalkers, nleaves, ndim_evolving = s_evol.shape
        _, _, _, ndim_stat = s_stat.shape

        cov_stat_xp = self.xp.asarray(self.cov_static)
        cov_evol_xp = self.xp.asarray(self.cov_evolving)

        q = {"static": s_stat.copy(), "evolving": s_evol.copy()}

        rng = random if not getattr(self, "use_gpu", False) else self.xp.random

        # Extract inverse temperatures (betas) for tempering
        # Fallback to an array of 1.0s if tempering is not initialized
        betas = self.xp.ones(ntemps)
        if hasattr(self, "temperature_control") and self.temperature_control is not None:
            betas = self.xp.asarray(self.temperature_control.betas)

        # Deterministic scheduling: 
        # A full cucle is all N blocks + 1 static update
        cycle_length = nleaves + 1
        # We perform module two for compatibility with other move classes with RedBlueMove which is called twice.
        current_target = (self.shared_state.step // 2) % cycle_length

        if current_target == nleaves:
            # Update the static parameters
            mean_stat = self.xp.zeros(len(self.cov_static))
            # Loop over temperatures to apply the scaled covariance matrix
            for t in range(ntemps):
                # Scale covariance by T = 1 / beta
                cov_t = cov_stat_xp / betas[t]
                
                static_step = rng.multivariate_normal(
                    mean_stat, cov_t, size=(nwalkers, 1)
                )
                q["static"][t] += static_step
        else:
            # Update the specific sequential leaf/Block
            leaf_idx = current_target
            mean_evol = self.xp.zeros(len(self.cov_evolving))
            # Loop over temperatures to apply the scaled covariance matrix
            for t in range(ntemps):
                # Scale covariance by T = 1 / beta
                cov_t = cov_evol_xp / betas[t]
                
                evolving_step = rng.multivariate_normal(
                    mean_evol, cov_t, size=nwalkers
                )
                q["evolving"][t, :, leaf_idx, :] += evolving_step

        # Increment the step counter for the next call of this move
        # We add by two for compatibility with other move classes with RedBlueMove which is called twice.
        self.shared_state.step += 2

        # symmetric proposal. log proposal ratio factor is 0
        factors = self.xp.zeros((ntemps,nwalkers))

        if getattr(self, "use_gpu", False) and not getattr(self, "return_gpu", False):
            q["static"] = q["static"].get()
            q["evolving"] = q["evolving"].get()
            factors = factors.get()

        return q, factors
    
class SequentialBlockedGibbsStretchMove(RedBlueMove):
    """Deterministic sequential Blocked Gibbs sampler using affine-invariant stretch proposals.
    Cycles through all evolving blocks and the static branch.  
    Both halves of the RedBlue ensemble update the same block within
    each full iteration (controlled via floor-division on ``SharedState.step``).
    Compatible with ``SequentialBlockedGibbsGaussianMove`` when both share the
    same ``SharedState`` instance.
    """

    def __init__(self, a=2.0, shared_state=None, **kwargs):
        """
        Custom RedBlue move class for deterministic, sequential Blocked updates 
        of the evolving parameters (leaves) and the hyper parameters (static branch).
        
        StretchMove is based on Goodman & Weare's affine-invariant move.
        
        a (float): stretch move scale parameter
        shared_state (SharedState): Object to synchronize the iteration step across mixed moves.
        """
        self.a = a
        # Fallback to an independent state if none is provided
        self.shared_state = shared_state if shared_state is not None else SharedState()
        super().__init__(**kwargs)
    
    def get_proposal(self, s, c, random, s_inds=None, c_inds=None, **kwargs):

        q = {}

        # s contains the ACTIVE walkers being moved
        # c contains the STATIONARY complementary walkers
        s_stat = self.xp.asarray(s["static"])
        s_evol = self.xp.asarray(s["evolving"])

        # the complementary walker sets must be flattened
        c_stat = self.xp.concatenate([self.xp.asarray(c_val) for c_val in c["static"]], axis=1)
        c_evol = self.xp.concatenate([self.xp.asarray(c_val) for c_val in c["evolving"]], axis=1)

        ntemps, nactive, nleaves, ndim_evol = s_evol.shape
        _, ncomp, _, ndim_stat = c_stat.shape

        q = {"static": s_stat.copy(), "evolving": s_evol.copy()}
        rng = random if not getattr(self, "use_gpu", False) else self.xp.random

        # Deterministic scheduling: N blocks + 1 static update
        cycle_length = nleaves + 1
        
        # Ensure both halves of the RedBlue split update the exact same block
        current_target = (self.shared_state.step // 2) % cycle_length

        # Draw random complementary walkers for each active walker
        rint = rng.randint(ncomp, size=(ntemps, nactive))

        # Draw the stretch scale variable z from the standard distribution
        zz = ((self.a - 1.0) * rng.rand(ntemps, nactive) + 1.0) ** 2.0 / self.a
        factors = self.xp.zeros((ntemps, nactive))

        # Setup advanced index for vectorization over temperatures
        t_idx = self.xp.arange(ntemps)[:, None]

        if current_target == nleaves:
            # ---------------- STATIC UPDATE ----------------
            # Extract values dynamically without slow nested for-loops
            c_val = c_stat[t_idx, rint, 0, :]
            s_val = s_stat[:, :, 0, :]
            
            # Expand dimensions to broadcast the scale factor across parameters
            zz_expanded = zz[:, :, None]
            
            # Propose new static parameters
            q["static"][:, :, 0, :] = c_val - zz_expanded * (c_val - s_val)
            
            # The factor scales by the dimensionality of the updated block
            factors += (ndim_stat - 1.0) * self.xp.log(zz)

        else:
            # ---------------- EVOLVING UPDATE ----------------
            leaf_idx = current_target
            
            c_val = c_evol[t_idx, rint, leaf_idx, :]
            s_val = s_evol[:, :, leaf_idx, :]
            zz_expanded = zz[:, :, None]
            
            q["evolving"][:, :, leaf_idx, :] = c_val - zz_expanded * (c_val - s_val)
            factors += (ndim_evol - 1.0) * self.xp.log(zz)

        # Increment the step counter for the next half-sweep
        self.shared_state.step += 1

        if getattr(self, "use_gpu", False) and not getattr(self, "return_gpu", False):
            q["static"] = q["static"].get()
            q["evolving"] = q["evolving"].get()
            factors = factors.get()

        return q, factors
    
class SequentialAdaptiveBlockedGibbsGaussianMove(RedBlueMove):
    """Deterministic sequential Blocked Gibbs sampler with an adaptive empirical-covariance kernel.
    Combines sequential block scheduling (via ``SharedState``) with an online
    covariance estimate built from the complementary walker ensemble.  A burn-in
    phase uses a heavily compressed scale factor for broad initial exploration,
    then switches to the theoretically optimal Gelman–Roberts–Gilks scaling
    ``(2.38^2 / d)`` once ``burn_in_steps`` iterations have elapsed.  A diagonal
    regularizer ``reg`` prevents covariance collapse during early burn-in when
    walkers are tightly clustered.
    """

    def __init__(self, mode_factor=None, burn_in_mode_factor=1e-4, burn_in_steps=1000, reg=1e-9, shared_state=None, **kwargs):
        """
        Custom RedBlueMove class for sequential adaptive updates of the
        evolving parameters (leaves) and hyper parameters (static branch).
        
        mode_factor (float): Scaling factor for the covariance. Defaults to the 
                             mathematically optimal (2.38^2 / d) for Random Walk.
        burn_in_mode_factor (float): Heavily compressed scale for early exploration.
        burn_in_steps (int): Number of MCMC iterations before switching to optimal scaling.
        reg (float): Small regularizer added to the diagonal to prevent matrix 
                     collapse during the very early burn-in steps.
        shared_state (SharedState): Object to synchronize the iteration step 
                                    across mixed moves.
        """
        self.mode_factor = mode_factor
        self.burn_in_mode_factor = burn_in_mode_factor
        self.burn_in_steps = burn_in_steps
        self.reg = reg
        self.shared_state = shared_state if shared_state is not None else SharedState()
        super().__init__(**kwargs)

    def get_proposal(self, s, c, random, s_inds=None, c_inds=None, **kwargs):

        q = {}

        # s contains the ACTIVE walkers being moved
        # c contains the STATIONARY complementary walkers (used to build the covariance)
        s_stat = self.xp.asarray(s["static"])
        s_evol = self.xp.asarray(s["evolving"])

        c_stat = self.xp.concatenate([self.xp.asarray(c_val) for c_val in c["static"]], axis=1)
        c_evol = self.xp.concatenate([self.xp.asarray(c_val) for c_val in c["evolving"]], axis=1)

        ntemps, nactive, nleaves, ndim_evol = s_evol.shape
        _, _, _, ndim_stat = s_stat.shape

        q = {"static": s_stat.copy(), "evolving": s_evol.copy()}
        rng = random if not getattr(self, "use_gpu", False) else self.xp.random

        # Deterministic scheduling: N blocks + 1 static update
        cycle_length = nleaves + 1
        
        # RedBlueMove calls get_proposal TWICE per full ensemble sweep. 
        # We use floor division (// 2) to ensure both halves of the ensemble 
        # update the exact same block before moving to the next one!
        current_target = (self.shared_state.step // 2) % cycle_length

        # Calculate the absolute MCMC iteration number
        # 1 full iteration = (cycle_length) blocks * 2 (RedBlue half-sweeps)
        current_mcmc_iteration = self.shared_state.step // (2 * cycle_length)

        # Apply the Dynamic Schedule
        if current_mcmc_iteration < self.burn_in_steps:
            base_scale = self.burn_in_mode_factor
        else:
            base_scale = self.mode_factor if self.mode_factor is not None else (2.38 ** 2)

        factors = self.xp.zeros((ntemps, nactive))

        # The eigh method for multivariate_normal covariance is only available in CuPy.
        mvn_kwargs = {"method": "eigh"} if getattr(self, "use_gpu", False) else {}

        if current_target == nleaves:
            # ---------------- STATIC UPDATE ----------------
            scale = base_scale / ndim_stat
            x_c = c_stat[:, :, 0, :] # Extract complementary walkers
            
            for t in range(ntemps):
                # 1. Calculate the empirical covariance of the COMPLEMENTARY walkers
                cov_t = self.xp.cov(x_c[t], rowvar=False) 
                # 2. Add regularization and optimally scale the jump
                cov_t = cov_t * scale + self.xp.eye(ndim_stat) * self.reg
                
                mean_stat = self.xp.zeros(ndim_stat)
                # ! Warning: "method" arg is unavailable if rng created from numpy!
                static_step = rng.multivariate_normal(mean_stat, cov_t, size=nactive, **mvn_kwargs) 
                q["static"][t, :, 0, :] += static_step

        else:
            # ---------------- EVOLVING UPDATE ----------------
            leaf_idx = current_target
            scale = base_scale / ndim_evol
            x_c = c_evol[:, :, leaf_idx, :]
            
            for t in range(ntemps):
                # 1. Calculate the empirical covariance of the COMPLEMENTARY walkers for this block
                cov_t = self.xp.cov(x_c[t], rowvar=False)
                # 2. Add regularization and optimally scale the jump
                cov_t = cov_t * scale + self.xp.eye(ndim_evol) * self.reg
                
                mean_evol = self.xp.zeros(ndim_evol)
                # ! Warning: "method" arg is unavailable if rng created from numpy!
                evolving_step = rng.multivariate_normal(mean_evol, cov_t, size=nactive, **mvn_kwargs)
                q["evolving"][t, :, leaf_idx, :] += evolving_step

        # Increment the step counter for the next half-sweep
        self.shared_state.step += 1

        if getattr(self, "use_gpu", False) and not getattr(self, "return_gpu", False):
            q["static"] = q["static"].get()
            q["evolving"] = q["evolving"].get()
            factors = factors.get()

        return q, factors