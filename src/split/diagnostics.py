import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from few.utils.constants import YRSID_SI
import os

def update_diagnostic_plots(sampler, diagnostics_dir, Nblocks, dt, slice_length,
                            ev_in_names, static_in_names, value_fixed_static, value_fixed_ev, 
                            indices_static_in, indices_ev_in, indices_static_fixed, indices_ev_fixed,
                            pars_names, true_pars, traj_indices, kerr_traj_instance,
                            val_samp_ev, val_samp_st, min_autocorr_iters):
    """Extract multi-branch chains, plot 1D walks, static posteriors, and t=0 projections.
    TODO:The input arguments list can probably be cleaned up."""
    print(f"\n[Step {sampler.iteration}] Generating diagnostic plots...")
    os.makedirs(diagnostics_dir, exist_ok=True)

    chain_static = sampler.get_chain()["static"][:, 0, :, 0, :] #chain_static has shape (nsteps, nwalkers, ndim_static).
    chain_evolving = sampler.get_chain()["evolving"][:, 0, :, :, :] #chain_evolving has shape (nsteps, nwalkers, Nblocks, ndim_evolving).

    current_nsteps, nwalkers, ndim_static = chain_static.shape
    ndim_evolving = chain_evolving.shape[-1]

    # only work with the final 10% of each chain.
    discard_idx = int(current_nsteps * 0.9)

    # Plot A1: Static Corner Plot
    flat_static = chain_static[discard_idx:].reshape(-1, ndim_static)
    try:
        fig_corner_st = corner.corner(flat_static, labels=static_in_names, truths=val_samp_st, show_titles=True)
        plt.savefig(f"{diagnostics_dir}/corner_static.png", bbox_inches='tight', dpi=300)
        plt.close(fig_corner_st)
    except ValueError as e:
        print(f" Skipping static corner plot: {e}")

    # Plot A2: Static 1D walks (includes burn-in)
    fig_1d_static, axs_static = plt.subplots(ndim_static, 1, figsize=(10, 3 * ndim_static))
    for j in range(ndim_static):
        axs_static[j].plot(chain_static[:,:,j], alpha=0.5)
        axs_static[j].axhline(val_samp_st[j], color='k', linestyle='--', label='True value')
        axs_static[j].set_ylabel(f"{static_in_names[j]}")
    axs_static[-1].set_xlabel("Steps")
    plt.savefig(f"{diagnostics_dir}/1dplots_static.png", bbox_inches='tight', dpi=300)
    plt.close(fig_1d_static)

    # Plot B: 1D walks and corner plots per block for evolving parameters
    for i in range(Nblocks):
        labels_ev_i = [f"{name}_{i}" for name in ev_in_names]
        # --- 1D Walks ---
        fig_1d, axs = plt.subplots(ndim_evolving, 1, figsize=(10, 3 * ndim_evolving))
        for j in range(ndim_evolving):
            axs[j].plot(chain_evolving[:, :, i, j], alpha=0.5)
            axs[j].axhline(val_samp_ev[i,j], color='k', linestyle='--', label='True value')
            axs[j].set_ylabel(labels_ev_i[j])

        axs[-1].set_xlabel("Steps")
        plt.savefig(f"{diagnostics_dir}/1dplots_block_{i}.png", bbox_inches='tight', dpi=300)
        plt.close(fig_1d)

        # --- Corner Plot ---
        flat_ev_i = chain_evolving[discard_idx:, :, i, :].reshape(-1, ndim_evolving)
        try:
            fig_corner_ev = corner.corner(flat_ev_i, labels=labels_ev_i, truths=val_samp_ev[i], show_titles=True)
            plt.savefig(f"{diagnostics_dir}/corner_evolving_{i}.png", bbox_inches='tight', dpi=300)
            plt.close(fig_corner_ev)
        except ValueError as e:
            print(f"Skipping evolving corner plot for block {i}: {e}")

    # Plot C: Autocorrelation Convergence
    if current_nsteps > 100:
        N_steps = np.exp(np.linspace(np.log(100), np.log(current_nsteps), 50)).astype(int)
        taus_static = np.empty((len(N_steps), ndim_static))
        taus_evolving = np.empty((len(N_steps), ndim_evolving))

        for idx, n in enumerate(N_steps):
            t_est_st = emcee.autocorr.integrated_time(chain_static[:n], tol=min_autocorr_iters, quiet=True)
            taus_static[idx] = t_est_st

            reshaped_ev = chain_evolving[:n].transpose(0,1,2,3).reshape(n, nwalkers*Nblocks, ndim_evolving)
            t_est_ev = emcee.autocorr.integrated_time(reshaped_ev, tol=min_autocorr_iters, quiet=True)
            taus_evolving[idx] = t_est_ev

        fig_ac, (ax1, ax2) = plt.subplots(2,1, figsize=(8,10))
        for j in range(ndim_static):
            ax1.loglog(N_steps, taus_static[:,j], "b-", alpha=0.3)
        ax1.loglog(N_steps, N_steps/50.0, "--r", label=r"$\tau = N/50$")
        ax1.set_title("Autocorrelations: Static branch")

        for j in range(ndim_evolving):
            ax2.loglog(N_steps, taus_evolving[:,j], "g-", alpha=0.3)
        ax2.loglog(N_steps, N_steps/50.0, "--r")
        ax2.set_title("Autocorrelations (max across blocks): evolving branch")

        ax1.legend()
        plt.savefig(f"{diagnostics_dir}/autocorr.png", bbox_inches='tight', dpi=300)
        plt.close(fig_ac)

    # Plot D: Backward Evolution to t=0
    if current_nsteps > 100:
        print("Evolving samples backwards to t=0 for joint posterior...")

        recent_static = chain_static[discard_idx:].reshape(-1, ndim_static)
        recent_evolving = chain_evolving[discard_idx:].reshape(-1, Nblocks, ndim_evolving)

        n_samples = len(recent_static)
        
        #rand_idx = np.random.choice(len(recent_static), size=n_samples, replace=False)

        samp_st = recent_static#[rand_idx]
        samp_ev = recent_evolving#[rand_idx]

        projected_t0_samples = []

        for j in range(n_samples):
            for i in range(Nblocks):
                T_backwards = (i * slice_length * dt) / YRSID_SI

                if T_backwards == 0.0:
                    projected_t0_samples.append(list(samp_st[j]) + list(samp_ev[j,i,:]))
                    continue

                pars_block = np.zeros(len(pars_names))
                pars_block[indices_static_in] = samp_st[j]
                pars_block[indices_ev_in] = samp_ev[j, i, :]
                pars_block[indices_static_fixed] = value_fixed_static
                pars_block[indices_ev_fixed] = value_fixed_ev[i, :]

                traj_args = pars_block[traj_indices]

                try:
                    traj_output = kerr_traj_instance(*traj_args[:-3], 
                                                     Phi_phi0=-traj_args[-3], #negative sign for the final phases so it is handled correctly in FEW
                                                     Phi_theta0=-traj_args[-2],
                                                     Phi_r0=-traj_args[-1], 
                                                     T=T_backwards, dt=dt, upsample=False, #upsample = False to get the final sample at exactly t=0.
                                                     integrate_backwards=True)
                    p0_proj = traj_output[1][-1]
                    e0_proj = traj_output[2][-1]
                    pp0_proj = -traj_output[4][0] % (2 * np.pi) #negative sign for phases so it is handled correctly in FEW, modulo for periodicity
                    pr0_proj = -traj_output[6][0] % (2 * np.pi)
                    
                    proj_ev_sampled = [p0_proj, e0_proj, pp0_proj, pr0_proj]
                    projected_t0_samples.append(list(samp_st[j]) + proj_ev_sampled)
                except Exception:
                    continue

        truths = [true_pars[i] for i in indices_static_in] + [true_pars[i] for i in indices_ev_in]
        projected_t0_samples = np.array(projected_t0_samples)

        if len(projected_t0_samples) > 0:
            labels_t0 = static_in_names + ev_in_names
            try:
                fig_t0 = corner.corner(projected_t0_samples, labels=labels_t0, truths=truths, show_titles=True)
                plt.savefig(f"{diagnostics_dir}/corner_t0_projected.png", bbox_inches='tight', dpi=300)
                plt.close(fig_t0)
            except ValueError:
                pass