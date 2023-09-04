import hydra
import os
import warnings

import pandas as pd

from omegaconf import DictConfig
from components._helpers import HuBERTModelForBenchmarking

# Ignore non-relevant warnings
warnings.filterwarnings("ignore", message="Applied workaround for CuDNN issue")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@hydra.main(version_base="1.3", config_path="./config", config_name="default")
def benchmark(cfg: DictConfig) -> None:
    
    for (name, value) in cfg.env.items():
        os.environ[name] = value

    model = HuBERTModelForBenchmarking(cfg)

    print(f"{cfg.encoder._target_}: {count_parameters(model):,} parameters")

    stats_df = model.benchmark()

    print(f"Throughput (forward+backward): {(stats_df.n_samples / ((stats_df.forward_ms + stats_df.backward_ms) / 1000)).mean():.2f} samples/s")
    print(f"Forward time in milliseconds, mean (std): {stats_df.forward_ms.mean():.2f} ({stats_df.forward_ms.std():.2f})")
    print(f"Backward time in milliseconds, mean (std): {stats_df.backward_ms.mean():.2f} ({stats_df.backward_ms.std():.2f})")

if __name__ == "__main__":
    benchmark()
