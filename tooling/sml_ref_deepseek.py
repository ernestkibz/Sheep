"""DeepSeek MoE v1 (deepseek-ai/deepseek-moe-16b, model_type "deepseek") as a reference for
sml_family_heads.py (2026-10-19).

transformers has no class for it: the checkpoint ships its own 2023 modeling_deepseek.py.
Its attention, norms, rotary and dense MLP are Llama's line for line, so the reference is
transformers' Llama with the MoE block below on every layer from first_k_dense_replace on
(every moe_layer_freq-th). The block is modeling_deepseek.py's DeepseekMoE and MoEGate at
inference: softmax scores, top-k (renormalized only when norm_topk_prob), the chosen
experts' outputs weighted and summed, plus the shared experts as one wider MLP. Parameter
names are the checkpoint's own (mlp.gate.weight, mlp.experts.N.*, mlp.shared_experts.*).
"""
import torch
import torch.nn.functional as F
from torch import nn

DEFAULTS = {"n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 2, "moe_intermediate_size": 128,
            "first_k_dense_replace": 1, "moe_layer_freq": 1, "norm_topk_prob": False, "scoring_func": "softmax"}


class DeepseekMLP(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    def __init__(self, hidden, n_experts):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_experts, hidden))


class DeepseekMoE(nn.Module):
    def __init__(self, hidden, moe):
        super().__init__()
        self.top_k, self.norm = moe["num_experts_per_tok"], moe["norm_topk_prob"]
        self.experts = nn.ModuleList([DeepseekMLP(hidden, moe["moe_intermediate_size"])
                                      for _ in range(moe["n_routed_experts"])])
        self.gate = MoEGate(hidden, moe["n_routed_experts"])
        self.shared_experts = DeepseekMLP(hidden, moe["moe_intermediate_size"] * moe["n_shared_experts"])

    def forward(self, x):
        shape = x.shape
        h = x.reshape(-1, shape[-1])
        scores = F.linear(h, self.gate.weight).softmax(dim=-1)
        w, idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.top_k > 1 and self.norm:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(h)
        for t in range(h.shape[0]):
            for j in range(self.top_k):
                y[t] += w[t, j] * self.experts[int(idx[t, j])](h[t])
        return y.reshape(shape) + self.shared_experts(x)


def convert(model, cfg_dict, over):
    """Swap the MoE layers of a transformers Llama in place; returns the deepseek config dict."""
    moe = dict(DEFAULTS, **{k: v for k, v in over.items() if k in DEFAULTS})
    hidden = model.config.hidden_size
    for i, layer in enumerate(model.model.layers):
        if i >= moe["first_k_dense_replace"] and i % moe["moe_layer_freq"] == 0:
            layer.mlp = DeepseekMoE(hidden, moe)
    out = dict(cfg_dict)
    out.update(moe, model_type="deepseek", architectures=["DeepseekForCausalLM"])
    out.pop("auto_map", None)
    return out
