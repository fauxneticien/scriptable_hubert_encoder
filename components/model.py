from . import verbatim_torchaudio as vta

def get_torchaudio_hubert_pretrain_base_encoder():
    encoder = vta._get_encoder(
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

    encoder.apply(vta._init_hubert_pretrain_model)

    return encoder
