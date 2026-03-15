import multiprocessing as mp
import os
import argparse
from split import SPLIT

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the SPLIT pipeline.")
    parser.add_argument('--emri', type=str, default='emri_config.json',
                        help='Path to the EMRI configuration JSON file.')
    parser.add_argument('--samp', type=str, default='sample_config.json',
                        help='Path to the sampling configuration JSON file.')
    parser.add_argument('--out', type=str, default='SPLIT_Outputs',
                        help='Path to output folder')
    args = parser.parse_args()

    # Mandatory initialization for PyTorch/CUDA multiprocessing safety
    mp.set_start_method('spawn', force=True)

    out_dir_root = "." #must already exist
    out_dir = os.path.join(out_dir_root, args.out)
    os.makedirs(out_dir, exist_ok=True)

    # Initialize the Orchestrator with the JSON configuration files
    sampler_pipeline = SPLIT(
        emri_config_path=args.emri,
        sample_config_path=args.samp,
        out_dir=out_dir
    )
    
    # Execute the three main stages of the pipeline sequentially
    sampler_pipeline.generate_injection_data() # you can provide a custom wavegen class and additional params here
    sampler_pipeline.build_priors()
    sampler_pipeline.run_sampler()