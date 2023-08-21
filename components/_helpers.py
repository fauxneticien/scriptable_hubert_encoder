import time
import torch

from pathlib import Path
from lhotse import CutSet

from lhotse.dataset import DynamicBucketingSampler
from lhotse.dataset.collation import collate_custom_field
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, encoder, masking_config):
        super().__init__()
        self.encoder = encoder
        self.encoder.to('cuda')

        self.masking_config = masking_config

    def forward(self, batch):
        start=time.time()

        B, T, C = batch['feats_padded'].shape

        padding_mask = _get_padding_mask(batch['feats_padded'], batch['feat_lens'])
        masks_for_modeling = _compute_mask_indices((B, T), padding_mask, **self.masking_config)

        # Zero-out random frames
        batch['feats_padded'][masks_for_modeling] = 0

        transformer_outputs = self.encoder(
            batch['feats_padded'].to('cuda'),
            batch['feat_lens'].to('cuda')
        )

        end=time.time()
        throughput= B/(end-start)

        return transformer_outputs, throughput

