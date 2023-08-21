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

    throughput_stats=[]

    for i in tqdm(range(cfg.warmup_iters), dynamic_ncols=True, desc="Running warmup..."):
        batch = next(iter(data.train_loader))
        _, _ = model(batch)

    for j in tqdm(range(cfg.num_iters), dynamic_ncols=True, desc="Benchmarking throughput..."):
        batch = next(iter(data.train_loader))
        _, throughput = model(batch)
        throughput_stats.append(throughput)

    # Calculate average without first warm-up step
    print(f"Throughput (data samples per second): {np.mean(throughput_stats[cfg.num_iters:]):.2f}")

if __name__ == "__main__":
    benchmark()
