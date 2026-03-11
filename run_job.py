import multiprocessing as mp
import os
from .src.split import split

if __name__ == '__main__':
    # Mandatory initialization for PyTorch/CUDA multiprocessing safety
    mp.set_start_method('spawn', force=True)

    out_dir_root = "/scratch/e1101888/HILIS/Paper_Examples/" #must already exist
    out_dir = os.path.join(out_dir_root, "Vanilla_EMRI_SemiCoherent_Outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Initialize the Orchestrator with the JSON configuration files
    sampler_pipeline = split(
        emri_config_path="emri_config.json",
        sample_config_path="sample_config.json",
        out_dir=out_dir
    )
    
    # Execute the three main stages of the pipeline sequentially
    sampler_pipeline.generate_injection_data() # you can provide a custom wavegen class and additional params here
    sampler_pipeline.build_priors()
    sampler_pipeline.run_sampler()