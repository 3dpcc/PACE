import torch.nn as nn
from mamba_ssm import Mamba


class MambaLayer(nn.Module):
    def __init__(self, ninp, layer_idx):
        super().__init__()

        self.mamba = Mamba(
            d_model=ninp,        # Model dimension d_model
            d_state=16,          # SSM state expansion factor
            d_conv=4,            # Local convolution width
            expand=2,            # Block expansion factor
            layer_idx=layer_idx, # Layer index
        )

        self.ln = nn.LayerNorm(ninp)

    def forward(self, src):
        out = self.mamba(self.ln(src))
        return out + src

    def step(self, src, infer_params):
        conv_state, ssm_state = self.mamba._get_states_from_cache(infer_params, src.size(0))
        out, _, _ = self.mamba.step(self.ln(src), conv_state, ssm_state)
        return out + src


class MambaBlock(nn.Module):
    def __init__(self, ninp, nlayers):
        super().__init__()

        self.layers = nn.ModuleList([
            MambaLayer(ninp, layer_idx)
            for layer_idx in range(nlayers)
        ])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)

        return src

    def step(self, src, infer_params_list):
        for layer, infer_params in zip(self.layers, infer_params_list):
            src = layer.step(src, infer_params)

        return src
