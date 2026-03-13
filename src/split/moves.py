from eryn.moves import MHMove, RedBlueMove
import numpy as np
import cupy as cp

#define a global custom move class for probabilistic blocked Gibbs updates of the evolving parameters.
class BlockedGibbsGaussianMove(MHMove):
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

        cov_stat_xp = self.xp.asarray(self.cov_static)
        cov_evol_xp = self.xp.asarray(self.cov_evolving)

        q = {"static": s_stat.copy(), "evolving": s_evol.copy()}

        rng = random if not getattr(self, "use_gpu", False) else self.xp.random

        #Randomly choose ONE branch to update
        if random.uniform() < self.prob_hyper:
            #update the hyper parameters (static branch).
            # generate steps for the static branch shape: (ntemps, nwalkers, 1 leaf, ndim_static)
            mean_stat = self.xp.zeros(len(self.cov_static))
            static_step = rng.multivariate_normal(
                mean_stat, cov_stat_xp, size=(ntemps, nwalkers, 1)
            )
            q["static"] += static_step

        else:
            #update ONE of the leaves (blocks) of the evolving branch.
            # randomly choose a leaf (block) to update. This probability can also be weighted if needed. 
            leaf_idx = random.choice(nleaves)

            # generate step for the evolving branch for the chosen leaf of shape (ntemps, nwalkers, ndim_evolving)
            mean_evol = self.xp.zeros(len(self.cov_evolving))
            evolving_step = rng.multivariate_normal(
                mean_evol, cov_evol_xp, size=(ntemps, nwalkers,)
            )
            q["evolving"][:, :, leaf_idx, :] += evolving_step

        # symmetric proposal. log proposal ratio factor is 0
        factors = self.xp.zeros((ntemps,nwalkers))

        if getattr(self, "use_gpu", False) and not getattr(self, "return_gpu", False):
            q["static"] = q["static"].get()
            q["evolving"] = q["evolving"].get()
            factors = factors.get()

        return q, factors

# similarly, define a probabilistic blocked stretch move class 
class BlockedStretchMove(RedBlueMove):
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

        if self.use_gpu and not getattr(self, "return_gpu", False):
            q["static"] = q["static"].get()
            q["evolving"] = q["evolving"].get()
            factors = factors.get()

        return q, factors
    
class SequentialBlockedGibbsGaussianMove(MHMove):
    def __init__(self, cov, **kwargs):
        """
        Custom MHmove class for sequential updates of the
        evolving parameters (leaves) and hyper parameters (static branch).
        cov (dict): keys: branch names values: covariance matrices for those branches.
        """
        self.cov_evolving = cov["evolving"]
        self.cov_static = cov["static"]

        # Internal step counter to track the Gibbs sweep
        self.step_counter = 0

        super().__init__(**kwargs)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        q = {}

        s_stat = self.xp.asarray(branches_coords["static"])
        s_evol = self.xp.asarray(branches_coords["evolving"])

        ntemps, nwalkers, nleaves, ndim_evolving = s_evol.shape

        cov_stat_xp = self.xp.asarray(self.cov_static)
        cov_evol_xp = self.xp.asarray(self.cov_evolving)

        q = {"static": s_stat.copy(), "evolving": s_evol.copy()}

        rng = random if not getattr(self, "use_gpu", False) else self.xp.random

        # Deterministic scheduling: 
        # A full cucle is all N blocks + 1 static update
        cycle_length = nleaves + 1
        current_target = self.step_counter % cycle_length

        if current_target == nleaves:
            # Update the static parameters
            mean_stat = self.xp.zeros(len(self.cov_static))
            static_step = rng.multivariate_normal(
                mean_stat, cov_stat_xp, size=(ntemps, nwalkers, 1)
            )
            q["static"] += static_step
        else:
            # Update the specific sequential leaf/Block
            leaf_idx = current_target
            mean_evol = self.xp.zeros(len(self.cov_evolving))
            evolving_step = rng.multivariate_normal(
                mean_evol, cov_evol_xp, size=(ntemps, nwalkers,)
            )
            q["evolving"][:, :, leaf_idx, :] += evolving_step

        # Increment the step counter for the next call of this move
        self.step_counter += 1

        # symmetric proposal. log proposal ratio factor is 0
        factors = self.xp.zeros((ntemps,nwalkers))

        if getattr(self, "use_gpu", False) and not getattr(self, "return_gpu", False):
            q["static"] = q["static"].get()
            q["evolving"] = q["evolving"].get()
            factors = factors.get()

        return q, factors
    
class SequentialBlockedStretchMove(RedBlueMove):
    def __init__(self, a=2.0, **kwargs):
        """
        Custom RedBlue move class for deterministic, sequential Blocked updates 
        of the evolving parameters (leaves) and the hyper parameters (static branch).
        
        StretchMove is based on Goodman & Weare's affine-invariant move.
        
        a (float): stretch move scale parameter
        """
        self.a = a
        self.step_counter = 0
        super().__init__(**kwargs)
    
    def get_proposal(self, s, c, random, s_inds=None, c_inds=None, **kwargs):
        use_gpu = getattr(self, "use_gpu", False)
        xp = cp if use_gpu else np

        q = {}

        # s contains the ACTIVE walkers being moved
        # c contains the STATIONARY complementary walkers
        s_stat = xp.asarray(s["static"])
        c_stat = xp.asarray(c["static"])
        s_evol = xp.asarray(s["evolving"])
        c_evol = xp.asarray(c["evolving"])

        ntemps, nactive, nleaves, ndim_evol = s_evol.shape
        _, ncomp, _, ndim_stat = c_stat.shape

        q = {"static": s_stat.copy(), "evolving": s_evol.copy()}
        rng = random if not use_gpu else xp.random

        # Deterministic scheduling: N blocks + 1 static update
        cycle_length = nleaves + 1
        
        # Ensure both halves of the RedBlue split update the exact same block
        current_target = (self.step_counter // 2) % cycle_length

        # Draw random complementary walkers for each active walker
        rint = rng.randint(ncomp, size=(ntemps, nactive))

        # Draw the stretch scale variable z from the standard distribution
        zz = ((self.a - 1.0) * rng.rand(ntemps, nactive) + 1.0) ** 2.0 / self.a
        factors = xp.zeros((ntemps, nactive))

        # Setup advanced index for vectorization over temperatures
        t_idx = xp.arange(ntemps)[:, None]

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
            factors += (ndim_stat - 1.0) * xp.log(zz)

        else:
            # ---------------- EVOLVING UPDATE ----------------
            leaf_idx = current_target
            
            c_val = c_evol[t_idx, rint, leaf_idx, :]
            s_val = s_evol[:, :, leaf_idx, :]
            zz_expanded = zz[:, :, None]
            
            q["evolving"][:, :, leaf_idx, :] = c_val - zz_expanded * (c_val - s_val)
            factors += (ndim_evol - 1.0) * xp.log(zz)

        # Increment the step counter for the next half-sweep
        self.step_counter += 1

        if use_gpu and not getattr(self, "return_gpu", False):
            q["static"] = q["static"].get()
            q["evolving"] = q["evolving"].get()
            factors = factors.get()

        return q, factors
    
class SequentialAdaptiveBlockedGibbsGaussianMove(RedBlueMove):
    def __init__(self, mode_factor=None, reg=1e-9, **kwargs):
        """
        Custom RedBlueMove class for sequential adaptive updates of the
        evolving parameters (leaves) and hyper parameters (static branch).
        
        mode_factor (float): Scaling factor for the covariance. Defaults to the 
                             mathematically optimal (2.38^2 / d) for Random Walk.
        reg (float): Small regularizer added to the diagonal to prevent matrix 
                     collapse during the very early burn-in steps.
        """
        self.mode_factor = mode_factor
        self.reg = reg
        self.step_counter = 0
        super().__init__(**kwargs)

    def get_proposal(self, s, c, random, s_inds=None, c_inds=None, **kwargs):
        use_gpu = getattr(self, "use_gpu", False)
        self.xp = cp if use_gpu else np

        q = {}

        # s contains the ACTIVE walkers being moved
        # c contains the STATIONARY complementary walkers (used to build the covariance)
        s_stat = self.xp.asarray(s["static"])
        s_evol = self.xp.asarray(s["evolving"])
        c_stat = self.xp.asarray(c["static"])
        c_evol = self.xp.asarray(c["evolving"])

        ntemps, nactive, nleaves, ndim_evol = s_evol.shape
        _, _, _, ndim_stat = s_stat.shape

        q = {"static": s_stat.copy(), "evolving": s_evol.copy()}
        rng = random if not getattr(self, "use_gpu", False) else self.xp.random

        # Deterministic scheduling: N blocks + 1 static update
        cycle_length = nleaves + 1
        
        # RedBlueMove calls get_proposal TWICE per full ensemble sweep. 
        # We use floor division (// 2) to ensure both halves of the ensemble 
        # update the exact same block before moving to the next one!
        current_target = (self.step_counter // 2) % cycle_length

        factors = self.xp.zeros((ntemps, nactive))

        if current_target == nleaves:
            # ---------------- STATIC UPDATE ----------------
            scale = self.mode_factor if self.mode_factor is not None else (2.38 ** 2) / ndim_stat
            x_c = c_stat[:, :, 0, :] # Extract complementary walkers
            
            for t in range(ntemps):
                # 1. Calculate the empirical covariance of the COMPLEMENTARY walkers
                cov_t = self.xp.cov(x_c[t], rowvar=False) 
                # 2. Add regularization and optimally scale the jump
                cov_t = cov_t * scale + self.xp.eye(ndim_stat) * self.reg
                
                mean_stat = self.xp.zeros(ndim_stat)
                static_step = rng.multivariate_normal(mean_stat, cov_t, size=nactive, method='eigh')
                q["static"][t, :, 0, :] += static_step

        else:
            # ---------------- EVOLVING UPDATE ----------------
            leaf_idx = current_target
            scale = self.mode_factor if self.mode_factor is not None else (2.38 ** 2) / ndim_evol
            x_c = c_evol[:, :, leaf_idx, :]
            
            for t in range(ntemps):
                # 1. Calculate the empirical covariance of the COMPLEMENTARY walkers for this block
                cov_t = self.xp.cov(x_c[t], rowvar=False)
                # 2. Add regularization and optimally scale the jump
                cov_t = cov_t * scale + self.xp.eye(ndim_evol) * self.reg
                
                mean_evol = self.xp.zeros(ndim_evol)
                evolving_step = rng.multivariate_normal(mean_evol, cov_t, size=nactive, method='eigh')
                q["evolving"][t, :, leaf_idx, :] += evolving_step

        # Increment the step counter for the next half-sweep
        self.step_counter += 1

        if getattr(self, "use_gpu", False) and not getattr(self, "return_gpu", False):
            q["static"] = q["static"].get()
            q["evolving"] = q["evolving"].get()
            factors = factors.get()

        return q, factors