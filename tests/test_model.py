import math

import pytest
import torch

from tnsr_core.config import ConfigError, ModelConfig
from tnsr_core.model import TNSRModel, causal_mask


def cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=16)
    base.update(kw)
    return ModelConfig(**base)


VARIANTS = [
    {},
    {"attn_impl": "manual"},
    {"pos_encoding": "rope"},
    {"norm": "layernorm", "bias": True},
    {"tie_embeddings": False},
]


@pytest.mark.parametrize("variant", VARIANTS)
def test_output_shapes_and_loss(variant):
    torch.manual_seed(0)
    model = TNSRModel(cfg(**variant))
    x = torch.randint(0, 50, (3, 10))
    logits, loss = model(x, x)
    assert logits.shape == (3, 10, 50)
    assert loss.ndim == 0
    # Near-uniform predictions at init: loss close to ln(vocab).
    assert abs(loss.item() - math.log(50)) < 0.5
    logits, loss = model(x)
    assert loss is None


def test_parameter_count_matches_formula():
    c = cfg()
    model = TNSRModel(c)
    d, v, t, L = c.n_embd, c.vocab_size, c.block_size, c.n_layer
    per_block = (
        (3 * d * d) + (d * d) + (4 * d * d) * 2 + 2 * d
    )  # qkv, proj, mlp, two rmsnorm weights
    expected = v * d + t * d + L * per_block + d  # tied lm_head, final norm
    assert model.num_parameters() == expected
    assert model.num_parameters(non_embedding=True) == expected - v * d - t * d


@pytest.mark.parametrize("variant", VARIANTS)
def test_causal_masking_future_tokens_do_not_affect_past(variant):
    torch.manual_seed(0)
    model = TNSRModel(cfg(**variant)).eval()
    x = torch.randint(0, 50, (1, 12))
    y = x.clone()
    y[0, 8:] = (y[0, 8:] + 7) % 50
    with torch.no_grad():
        lx, _ = model(x)
        ly, _ = model(y)
    assert torch.allclose(lx[0, :8], ly[0, :8], atol=1e-5)
    assert not torch.allclose(lx[0, 8:], ly[0, 8:], atol=1e-5)


def test_sdpa_and_manual_attention_agree():
    torch.manual_seed(0)
    a = TNSRModel(cfg(attn_impl="sdpa")).eval()
    b = TNSRModel(cfg(attn_impl="manual")).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 50, (2, 16))
    with torch.no_grad():
        assert torch.allclose(a(x)[0], b(x)[0], atol=1e-5)


def test_causal_mask_with_offset():
    m = causal_mask(2, 5, torch.device("cpu"))
    assert m.tolist() == [[True, True, True, True, False], [True, True, True, True, True]]


@pytest.mark.parametrize("variant", [{}, {"attn_impl": "manual"}, {"pos_encoding": "rope"}])
def test_kv_cache_matches_full_forward(variant):
    torch.manual_seed(0)
    model = TNSRModel(cfg(**variant)).eval()
    x = torch.randint(0, 50, (1, 12))
    with torch.no_grad():
        full, _ = model(x)
        cache = model.new_cache()
        out = [model(x[:, :5], cache=cache)[0]]
        out.append(model(x[:, 5:8], cache=cache)[0])  # multi-token chunk with past
        for i in range(8, 12):
            out.append(model(x[:, i : i + 1], cache=cache)[0])
    assert torch.allclose(full, torch.cat(out, dim=1), atol=1e-5)


def test_gradients_reach_every_parameter():
    torch.manual_seed(0)
    model = TNSRModel(cfg(pos_encoding="learned"))
    x = torch.randint(0, 50, (2, 16))
    _, loss = model(x, x)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_overfits_a_tiny_batch():
    torch.manual_seed(0)
    model = TNSRModel(cfg(n_embd=32))
    x = torch.randint(0, 50, (2, 16))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(150):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


def test_rejects_sequences_longer_than_block_size():
    model = TNSRModel(cfg())
    with pytest.raises(ValueError, match="block_size"):
        model(torch.zeros(1, 17, dtype=torch.long))


def test_last_only_logits():
    model = TNSRModel(cfg()).eval()
    x = torch.randint(0, 50, (1, 9))
    with torch.no_grad():
        full, _ = model(x)
        last, _ = model(x, last_only=True)
    assert last.shape == (1, 1, 50)
    assert torch.allclose(full[:, -1:], last, atol=1e-6)


@pytest.mark.parametrize(
    "bad",
    [
        {"n_embd": 30, "n_head": 4},
        {"vocab_size": 1},
        {"dropout": 1.0},
        {"norm": "batchnorm"},
        {"pos_encoding": "alibi"},
        {"n_layer": 0},
        {"attn_impl": "flash"},
    ],
)
def test_invalid_model_config(bad):
    with pytest.raises(ConfigError):
        TNSRModel(cfg(**bad))


def test_weight_tying_and_decay_groups():
    model = TNSRModel(cfg())
    assert model.lm_head.weight is model.tok_emb.weight
    groups = model.optimizer_groups(0.1)
    n_params = sum(p.numel() for g in groups for p in g["params"])
    assert n_params == model.num_parameters()
    assert all(p.dim() < 2 for p in groups[1]["params"])
