from .GlobalFusionClassifier import GlobalFusionClassifier
from .LocalFusionClassifier import LocalFusionClassifier
import torch
from torch import nn


class Gate_fusion(nn.Module):
    """HTER evidence calibration and bounded disagreement-aware routing.

    The module keeps the dual evidence paths intact, calibrates global/local
    evidence with SAE and LBE, and then applies a budgeted HDE-RER gate shift.
    """

    def __init__(self, args):
        super().__init__()

        self.Global_path = GlobalFusionClassifier(args)
        self.Local_path = LocalFusionClassifier(args)

        model_args = args.model
        dim = model_args.proj_dst_dim

        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1),
        )

        self.regression = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        self.sae_scale = getattr(model_args, "sae_support_scale", 0.10)
        self.lbe_scale = getattr(model_args, "lbe_behavior_scale", 0.10)
        self.sae_support = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
        )
        self.lbe_behavior = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.sae_support[-1].weight)
        nn.init.zeros_(self.sae_support[-1].bias)
        nn.init.zeros_(self.lbe_behavior[-1].weight)
        nn.init.zeros_(self.lbe_behavior[-1].bias)

        obs_dim = dim * 9
        self.hde_observer = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
        )
        self.gate_delta = nn.Linear(dim // 2, 2)
        nn.init.zeros_(self.gate_delta.weight)
        nn.init.zeros_(self.gate_delta.bias)

        self.route_probe = nn.Linear(dim // 2, 3)
        nn.init.zeros_(self.route_probe.weight)
        nn.init.zeros_(self.route_probe.bias)

        self.max_gate_shift = getattr(model_args, "hde_max_gate_shift", 0.15)
        self.avg_gate_shift = torch.tensor(0.0)
        self.avg_route_entropy = torch.tensor(0.0)

    def _build_hde_observation(self, modal):
        text = modal["text"]
        visual = modal["visual"]
        audio = modal["audio"]
        return torch.cat(
            [
                text,
                visual,
                audio,
                torch.abs(text - visual),
                torch.abs(text - audio),
                torch.abs(visual - audio),
                text * visual,
                text * audio,
                visual * audio,
            ],
            dim=-1,
        )

    def _hder_modulated_gate(self, global_feat, local_feat, modal):
        base_gate = self.fusion_gate(torch.cat([global_feat, local_feat], dim=1))
        obs = self._build_hde_observation(modal)
        hidden = self.hde_observer(obs)
        delta = self.max_gate_shift * torch.tanh(self.gate_delta(hidden))
        final_gate = torch.softmax(torch.log(torch.clamp(base_gate, min=1e-6)) + delta, dim=-1)

        route_weights = torch.softmax(self.route_probe(hidden), dim=-1)
        self.avg_gate_shift = delta.detach().abs().mean()
        self.avg_route_entropy = (
            -(route_weights.detach() * torch.log(torch.clamp(route_weights.detach(), min=1e-6))).sum(dim=-1).mean()
        )

        return final_gate, {"route_weights": route_weights, "gate_delta": delta}

    def _enhance_evidence(self, global_feat, local_feat, modal):
        text = modal["text"]
        visual = modal["visual"]
        audio = modal["audio"]
        av = 0.5 * (audio + visual)

        semantic_input = torch.cat([text, av, torch.abs(text - av)], dim=-1)
        semantic_support = self.sae_scale * torch.tanh(self.sae_support(semantic_input))
        global_feat = global_feat * (1.0 + semantic_support)

        behavior_input = torch.cat([audio, visual, torch.abs(audio - visual)], dim=-1)
        behavior_residual = self.lbe_scale * torch.tanh(self.lbe_behavior(behavior_input))
        local_feat = local_feat + behavior_residual

        return global_feat, local_feat

    def forward(self, video, audio, text):
        global_feat, global_modal = self.Global_path(video, audio, text, return_modal=True)
        local_feat, local_modal = self.Local_path(video, audio, text, return_modal=True)

        modal = {
            "text": 0.5 * (global_modal["text"] + local_modal["text"]),
            "visual": 0.5 * (global_modal["visual"] + local_modal["visual"]),
            "audio": 0.5 * (global_modal["audio"] + local_modal["audio"]),
        }

        global_feat, local_feat = self._enhance_evidence(global_feat, local_feat, modal)
        gate, scores = self._hder_modulated_gate(global_feat, local_feat, modal)
        fused = gate[:, 0:1] * global_feat + gate[:, 1:2] * local_feat
        pred = self.regression(fused)

        return pred


def build_model(args):
    model = Gate_fusion(args)

    return model
