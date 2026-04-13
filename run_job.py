import multiprocessing as mp
import os
import argparse

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run the SPLIT pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--emri', type=str, default='emri_config.json',
                        help="Path to the EMRI configuration JSON file.\n"
                        "Allowed keywords:\n"
                        "  - analysis setup: data_model, analysis_model, response, add_noise, rng_seed\n"
                        "  - EMRI Params: m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0\n"
                        "  - Observation: T, dt, desired_SNR\n"
                        "  - Custom injection: data_add_args, analysis_add_args\n"
                        "  - Blocks: Nblocks, alpha_block\n"
                        "  - Frequency mask: fmin, fmax\n"
                        "  - Likelihood/Prior: nu_like, nu_prior, sigma_prior")
    
    parser.add_argument('--samp', type=str, default='sample_config.json',
                        help="Path to the sampling configuration JSON file.\n"
                             "Allowed keywords:\n"
                             "  - Param Buckets: evolving_params, static_params, fixed_evolving\n"
                             "  - Sampler Setup: jitter, nwalkers, ntemps, nsteps, burn, thin_by, discard\n"
                             "  - Logistics: check_interval, check_converge\n"
                             "  - Moves: moves, adapt_burn_steps, burn_in_mode_factor\n"
                             "  - File I/O: filename (or new_filename), old_filename")
    
    parser.add_argument('--out', type=str, default='SPLIT_Outputs',
                        help="Path to the root output folder where chains and diagnostics will be saved.")
    
    args = parser.parse_args()

    # Mandatory initialization for PyTorch/CUDA multiprocessing safety
    mp.set_start_method('spawn', force=True)

    # Automatically set the root directory to the current working directory
    out_dir_root = os.getcwd()
    out_dir = os.path.join(out_dir_root, args.out)
    os.makedirs(out_dir, exist_ok=True)

    # Initialize the Orchestrator with the JSON configuration files
    from split import SPLIT

    sampler_pipeline = SPLIT(
        emri_config_path=args.emri,
        sample_config_path=args.samp,
        out_dir=out_dir
    ) # you can provide a custom wavegen class and additional params here
    
    # Execute the three main stages of the pipeline sequentially
    sampler_pipeline.generate_injection_data() 
    sampler_pipeline.build_priors()
    sampler_pipeline.run_sampler()