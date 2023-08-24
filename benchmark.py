import hydra
import os

import numpy as np

from omegaconf import DictConfig
from components._helpers import HuBERTModelForBenchmarking
from tqdm import tqdm

@hydra.main(version_base="1.3", config_path="./config", config_name="default")
def benchmark(cfg: DictConfig) -> None:
    
    for (name, value) in cfg.env.items():
        os.environ[name] = value

    data = hydra.utils.instantiate(cfg.data)

    model = HuBERTModelForBenchmarking(
        encoder = hydra.utils.instantiate(cfg.encoder),
        masking_config = cfg.masking
    )

    total_iters = cfg.num_iters + cfg.warmup_iters

    throughput_stats=[]

    for i, batch in (pbar := tqdm(enumerate(data.train_loader), desc="Running warmup", total = total_iters)):

        if i == total_iters:
            break

        _, throughput = model(batch)

        if i == cfg.warmup_iters:
            pbar.set_description("Benchmarking throughput")

        if i >= cfg.warmup_iters:
            throughput_stats.append(throughput)

    # Calculate average without first warm-up step
    print(f"Throughput (data samples per second): {np.mean(throughput_stats):.2f}")

if __name__ == "__main__":
    benchmark()
