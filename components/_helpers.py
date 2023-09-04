import hydra
import time
import torch

from pathlib import Path
from lhotse import CutSet

from lhotse.dataset import DynamicBucketingSampler
from lhotse.dataset.collation import collate_custom_field
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .verbatim_torchaudio import _get_padding_mask, _compute_mask_indices

class HuBERTPretrainingDataset(Dataset):
    def __getitem__(self, cuts: CutSet) -> dict:
        cuts = cuts.sort_by_duration()

        # Collate and pad
        feats_padded, feat_lens = collate_custom_field(cuts, 'fbank', pad_value=0)
        # Note we'll use a negative integer for padding the labels since 0 is a valid label
        ptlabels_padded, ptlabels_lengths = collate_custom_field(cuts, 'ptlabel', pad_value=-100)

        return {"feats_padded": feats_padded, "feat_lens": feat_lens, "ptlabels_padded": ptlabels_padded, "ptlabels_lengths": ptlabels_lengths}

class MiniLibriSpeech:

    def __init__(self, train_data_dir, sampler_kwargs, dloader_kwargs):

        self.train_data_dir = Path(train_data_dir)

        self.train_cuts = CutSet.from_shar(fields={
            'cuts': sorted(list(self.train_data_dir.glob("cuts.*.jsonl.gz"))),
            'fbank': sorted(list(self.train_data_dir.glob("fbank.*.tar"))),
            'ptlabel': sorted(list(self.train_data_dir.glob("ptlabel.*.tar")))
        })

        self.train_sampler = DynamicBucketingSampler(self.train_cuts, **sampler_kwargs)

        self.train_loader = DataLoader(
            HuBERTPretrainingDataset(),
            sampler=self.train_sampler,
            batch_size=None,
            **dloader_kwargs
        )

class HuBERTModelForBenchmarking(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg

        self.encoder = hydra.utils.instantiate(cfg.encoder)
        self.final_proj = torch.nn.Linear(cfg.encoder.embed_dim, cfg.num_pretraining_targets)

        self.loss = hydra.utils.instantiate(cfg.loss)

        self.optimizer = hydra.utils.instantiate(cfg.optim.optimizer, params=self.parameters())
        self.scheduler = hydra.utils.instantiate(cfg.optim.scheduler, optimizer=self.optimizer)

        self.encoder.to('cuda')
        self.final_proj.to('cuda')
        self.loss.to('cuda')

    def forward(self, batch, skip_mask=False, skip_nomask=True):

        B, T, C = batch['feats_padded'].shape

        padding_mask = _get_padding_mask(batch['feats_padded'], batch['feat_lens'])
        masks_for_modeling = _compute_mask_indices((B, T), padding_mask, **self.cfg.masking)

        # Zero-out random frames
        batch['feats_padded'][masks_for_modeling] = 0

        self.optimizer.zero_grad()

        start = time.time()
        transformer_outputs = self.encoder(
            batch['feats_padded'].to('cuda'),
            batch['feat_lens'].to('cuda')
        )

        mask_m = torch.logical_and(~padding_mask, masks_for_modeling)
        logits_m = self.final_proj(transformer_outputs[mask_m]) if not skip_mask else None
        logits_u = self.final_proj(transformer_outputs[~mask_m]) if not skip_nomask else None
        forward_end=time.time()

        loss = self.loss(logits_m, logits_u, masks_for_modeling, batch["ptlabels_padded"].to('cuda'))
        loss.backward()
        backward_end=time.time()

        self.optimizer.step()
        self.scheduler.step()

        return {
            'n_samples': B,
            'forward': forward_end-start,
            'backward': backward_end-forward_end
        }

    def benchmark(self):

        data = hydra.utils.instantiate(self.cfg.data)

        total_iters = self.cfg.num_iters + self.cfg.warmup_iters

        throughput_stats=[]

        for i, batch in (pbar := tqdm(enumerate(data.train_loader), desc="Running warmup", total = total_iters, dynamic_ncols=True)):

            if i == total_iters:
                break

            iter_stats = self.forward(batch)

            if i == self.cfg.warmup_iters:
                pbar.set_description("Benchmarking throughput")

            if i >= self.cfg.warmup_iters:
                throughput_stats.append(iter_stats)

        return throughput_stats
