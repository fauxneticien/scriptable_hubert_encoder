import torch

from . import verbatim_torchaudio as vta
from . import scriptable as sta

class ScriptableTorchAudioEncoder(torch.nn.Module):
    def __init__(self, **encoder_kwargs):
        super().__init__()
        self.encoder = sta._get_encoder(**encoder_kwargs)
        self.encoder.apply(sta._init_hubert_pretrain_model)

    def forward(self, feats_padded, feat_lens):
        return self.encoder(feats_padded, feat_lens)

class VanillaTorchAudioEncoder(torch.nn.Module):
    def __init__(self, **encoder_kwargs):
        super().__init__()
        self.encoder = vta._get_encoder(**encoder_kwargs)
        self.encoder.apply(vta._init_hubert_pretrain_model)

    def forward(self, feats_padded, feat_lens):
        return self.encoder(feats_padded, feat_lens)
        
def get_torchaudio_hubert_pretrain_base_encoder():
    encoder = VanillaTorchAudioEncoder(
        in_features=80,
        embed_dim=768,
        dropout_input=0.1,
        pos_conv_kernel=128,
        pos_conv_groups=16,
        num_layers=12,
        num_heads=12,
        attention_dropout=0.1,
        ff_interm_features=3072,
        ff_interm_dropout=0.0,
        dropout=0.1,
        layer_norm_first=False,
        layer_drop=0.05,
    )

    return encoder.encoder
