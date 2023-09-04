# scriptable_hubert_encoder

This repository aims to create a scriptable HuBERT encoder to examine if/how well [cramming strategies](https://github.com/JonasGeiping/cramming) proposed for the original BERT encoder can help reduce the compute footprint needed to pretrain HuBERT models for low-resource scenarios.

# Setup

## Colab

To setup on Colab, run the code below in a code block:

```
%%capture
%%bash

# Setup Colab
if [ -n "$COLAB_RELEASE_TAG" ]; then
  rm -rf ./*
  curl -f https://raw.githubusercontent.com/fauxneticien/scriptable_hubert_encoder/main/setup-colab.sh | bash
fi
```

## Local (conda)

```
# Create new conda environment in ./env
conda create -y --prefix ./env python=3.10 --no-default-packages

# Activate envionrment
conda activate ./env

# Run setup
bash setup-local-conda.sh
```

# Run benchmark

```
# Default
python benchmark.py

# Some config in config/encoder
python benchmark.py encoder=crammed
```
