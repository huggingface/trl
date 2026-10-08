import copy
from functools import partial

import pytest
import torch
import torch.nn.functional as F

from trl.kernels.tree_gated_delta_rule import (
    TreeGDNPlan,
    reference_tree_gated_delta_rule,
    tree_causal_conv1d,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Tree GDN requires a CUDA GPU")


def paths(offsets, parents):
    rows = []
    for s, p in enumerate(parents):
        rows.append(([] if p == -1 else rows[p]) + list(range(offsets[s], offsets[s + 1])))
    return [r for s, r in enumerate(rows) if s not in parents]


def recurrent(q, k, v, g, beta):
    q, k = [x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6) for x in (q, k)]
    q = q / q.shape[-1] ** 0.5
    q, k = [x.repeat_interleave(v.shape[2] // x.shape[2], dim=2) for x in (q, k)]
    state = v.new_zeros((1, v.shape[2], k.shape[-1], v.shape[-1]), dtype=torch.float32)
    out = []
    for t in range(q.shape[1]):
        state = state * g[:, t].float().exp()[..., None, None]
        delta = (v[:, t].float() - (state * k[:, t, :, :, None]).sum(-2)) * beta[:, t, :, None].float()
        state = state + k[:, t, :, :, None] * delta[..., None, :]
        out.append((state * q[:, t, :, :, None]).sum(-2))
    return torch.stack(out, dim=1)


def reference_chunk(q, k, v, g, beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True):
    """Independent ordinary-sequence oracle; fp32 recurrence with the model's output dtype."""
    assert initial_state is None and not output_final_state and use_qk_l2norm_in_kernel
    return recurrent(q, k, v, g, beta).to(v.dtype), None


def rechunked_fla(q, k, v, g, beta, *, cut, **kwargs):
    """Diagnostic: ordinary independent sequences, but with a state handoff at the tree's fork boundary."""
    fla = pytest.importorskip("fla.ops.gated_delta_rule")
    kwargs.pop("initial_state", None)
    kwargs.pop("output_final_state", None)
    prefix, state = fla.chunk_gated_delta_rule(
        q[:, :cut], k[:, :cut], v[:, :cut], g[:, :cut], beta[:, :cut], output_final_state=True, **kwargs
    )
    suffix, _ = fla.chunk_gated_delta_rule(
        q[:, cut:], k[:, cut:], v[:, cut:], g[:, cut:], beta[:, cut:], initial_state=state, **kwargs
    )
    return torch.cat((prefix, suffix), dim=1), None


@pytest.fixture(params=["reference", "vendored", "vendored_aligned"])
def model_kernel(request, monkeypatch):
    if request.param == "reference":
        from trl.experimental.async_grpo.tree import gdn

        monkeypatch.setattr(gdn, "tree_chunk_gated_delta_rule", reference_tree_gated_delta_rule)
        return reference_chunk
    if request.param == "vendored_aligned":
        return rechunked_fla
    return pytest.importorskip("fla.ops.gated_delta_rule").chunk_gated_delta_rule


@pytest.mark.parametrize("head_dim", [32, 128])
def test_reference_triton_against_compiled_scan(head_dim):
    from benchmarks.tree_gdn_torch import torch_tree_gdn

    torch.manual_seed(4)
    prefix, suffix, branches = 65, 129, 2
    n = prefix + branches * suffix
    plan = TreeGDNPlan.build((0, prefix, prefix + suffix, n), (-1, 0, 0), "cuda")
    data = [torch.randn(1, n, 2, head_dim, device="cuda") for _ in range(3)]
    data += [-torch.rand(1, n, 2, device="cuda"), torch.rand(1, n, 2, device="cuda")]
    actual = [x.requires_grad_() for x in data]
    expected = [x.detach().clone().requires_grad_() for x in data]
    a = reference_tree_gated_delta_rule(*actual, plan)
    b = torch_tree_gdn(*expected, plan, prefix, branches)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
    weight = torch.randn_like(a)
    for ga, gb in zip(
        torch.autograd.grad((a * weight).sum(), actual), torch.autograd.grad((b * weight).sum(), expected), strict=True
    ):
        torch.testing.assert_close(ga, gb, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("prefix", [1, 3, 63, 64, 65, 129])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("decay_scale", [0.2, 20.0])
def test_reference_outputs_and_gradients(prefix, dtype, decay_scale, record_property):
    torch.manual_seed(12)
    # Nested fork, short segments, unequal lengths, and an independent root.
    offsets = (0, prefix, prefix + 2, prefix + 5, prefix + 9, prefix + 14, prefix + 17)
    parents = (-1, 0, 1, 1, 0, -1)
    plan = TreeGDNPlan.build(offsets, parents, "cuda")
    n = offsets[-1]
    data = [torch.randn(1, n, 2, 32, device="cuda", dtype=dtype) for _ in range(2)]
    data += [torch.randn(1, n, 4, 32, device="cuda", dtype=dtype)]
    data += [-torch.rand(1, n, 4, device="cuda") * decay_scale, torch.rand(1, n, 4, device="cuda", dtype=dtype)]
    actual = [x.detach().requires_grad_() for x in data]
    expected = [x.detach().clone().requires_grad_() for x in data]
    out = reference_tree_gated_delta_rule(*actual, plan)
    loss, ref_loss = 0, 0
    for row in paths(offsets, parents):
        ref = recurrent(*(x[:, row] for x in expected))
        weight = torch.randn_like(ref)
        torch.testing.assert_close(out[:, row].float(), ref, atol=0.007, rtol=0.03)
        loss = loss + (out[:, row].float() * weight).sum()
        ref_loss = ref_loss + (ref * weight).sum()
    grads = torch.autograd.grad(loss, actual)
    refs = torch.autograd.grad(ref_loss, expected)
    max_error = 0
    for a, b in zip(grads, refs, strict=True):
        torch.testing.assert_close(a.float(), b.float(), atol=0.035, rtol=0.06)
        relative_rms = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)
        assert relative_rms < (0.002 if dtype == torch.float32 else 0.02), relative_rms.item()
        max_error = max(max_error, relative_rms.item())
    record_property("max_gradient_relative_rms", max_error)


def test_reference_no_branch_and_fanout():
    for parents in [(-1,), (-1,) + (0,) * 8]:
        offsets = tuple(65 * s for s in range(len(parents) + 1))
        plan = TreeGDNPlan.build(offsets, parents, "cuda")
        torch.manual_seed(1)
        n = offsets[-1]
        data = [torch.randn(1, n, 2, 32, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        data += [-torch.rand(1, n, 2, device="cuda") * 0.01, torch.rand(1, n, 2, device="cuda")]
        actual = [x.requires_grad_() for x in data]
        expected = [x.detach().clone().requires_grad_() for x in data]
        out = reference_tree_gated_delta_rule(*actual, plan)
        row_index = torch.tensor([t for row in paths(offsets, parents) for t in row], device="cuda")
        ref = torch.cat([recurrent(*(x[:, row] for x in expected)) for row in paths(offsets, parents)], dim=1)
        ref = ref.to(out.dtype)
        selected = out.index_select(1, row_index)
        torch.testing.assert_close(selected, ref, atol=0.01, rtol=0.03)
        weight = torch.randn_like(ref)
        ga = torch.autograd.grad((selected * weight).sum(), actual)
        gb = torch.autograd.grad((ref * weight).sum(), expected)
        for a, b in zip(ga, gb, strict=True):
            torch.testing.assert_close(a, b, atol=0.035, rtol=0.06)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_convolution(dtype):
    offsets, parents = (0, 2, 3, 5, 6, 9), (-1, 0, 1, 0, -1)
    plan = TreeGDNPlan.build(offsets, parents, "cuda")
    torch.manual_seed(0)
    x = torch.randn(1, 9, 8, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(8, 1, 4, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(8, device="cuda", dtype=dtype, requires_grad=True)
    refs = [t.detach().clone().requires_grad_() for t in (x, w, b)]
    out = tree_causal_conv1d(x, w, b, plan)
    loss, ref_loss = 0, 0
    for row in paths(offsets, parents):
        ref = F.silu(F.conv1d(refs[0][:, row].transpose(1, 2), refs[1], refs[2], padding=3, groups=8)[..., : len(row)])
        torch.testing.assert_close(out[:, row], ref.transpose(1, 2), atol=0.02, rtol=0.02)
        loss = loss + out[:, row].float().square().sum()
        ref_loss = ref_loss + ref.float().square().sum()
    for actual, expected in zip(
        torch.autograd.grad(loss, (x, w, b)), torch.autograd.grad(ref_loss, refs), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=0.25 if dtype == torch.bfloat16 else 1e-4, rtol=0.025)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen_gdn_layer(dtype, record_property, model_kernel):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    from trl.experimental.async_grpo.tree.gdn import qwen3_5_tree_gdn

    torch.manual_seed(3)
    config = Qwen3_5TextConfig(
        hidden_size=128,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        dtype=dtype,
    )
    with torch.device("cuda"):
        layer = Qwen3_5GatedDeltaNet(config, 0).to(dtype)
    baseline = copy.deepcopy(layer)
    baseline.chunk_gated_delta_rule = partial(model_kernel, cut=65) if model_kernel is rechunked_fla else model_kernel
    offsets, parents = (0, 65, 71, 80), (-1, 0, 0)
    plan = TreeGDNPlan.build(offsets, parents, "cuda")
    x = torch.randn(1, 80, 128, device="cuda", dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    out = qwen3_5_tree_gdn(layer, x, plan)
    loss, ref_loss = 0, 0
    rows = paths(offsets, parents)
    max_length = max(map(len, rows))
    row_index = torch.tensor([row + [80] * (max_length - len(row)) for row in rows], device="cuda")
    ref_batch = baseline(F.pad(y, (0, 0, 0, 1))[0, row_index])
    for r, row in enumerate(rows):
        ref = ref_batch[r : r + 1, : len(row)]
        torch.testing.assert_close(out[:, row], ref, atol=0.025, rtol=0.04)
        weight = torch.randn_like(ref)
        loss = loss + (out[:, row].float() * weight).sum()
        ref_loss = ref_loss + (ref.float() * weight).sum()
    loss.backward()
    ref_loss.backward()
    max_error = 0
    for a, b in [(x, y), *zip(layer.parameters(), baseline.parameters(), strict=True)]:
        torch.testing.assert_close(a.grad, b.grad, atol=0.25, rtol=0.06)
        relative_rms = (a.grad.float() - b.grad.float()).norm() / b.grad.float().norm().clamp_min(1e-8)
        assert relative_rms < 0.04, (relative_rms.item(), a.grad.norm().item(), b.grad.norm().item())
        max_error = max(max_error, relative_rms.item())
    record_property("max_gradient_relative_rms", max_error)


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hybrid_qwen_model(checkpointing, dtype, record_property, model_kernel):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from trl.experimental.async_grpo.packing import TreePacking
    from trl.experimental.async_grpo.tree import PrefixForest, register_tree_attention
    from trl.experimental.async_grpo.tree.gdn import enable_qwen3_5_tree_gdn
    from trl.trainer.utils import add_fused_lm_head

    torch.manual_seed(8)
    register_tree_attention()
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        dtype=dtype,
        use_cache=False,
        attn_implementation="sdpa",
    )
    with torch.device("cuda"):
        baseline = Qwen3_5ForCausalLM(config).to(dtype).train()
    for layer in baseline.model.layers:
        if layer.block_type == "linear_attention":
            layer.linear_attn.chunk_gated_delta_rule = model_kernel
    model = copy.deepcopy(baseline)
    model.set_attn_implementation("tree_flex")
    enable_qwen3_5_tree_gdn(model)
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        baseline.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        add_fused_lm_head(model)
    # Uses the real (token, next-token) trie and an independent second tree.
    rows = [list(range(1, 18)) + [20, 21, 22], list(range(1, 18)) + [25, 26], [40, 41, 42]]
    forest = PrefixForest()
    for row in rows:
        forest.insert(row, 0)
    packed, layout, mapping = forest.linearize()
    if model_kernel is rechunked_fla:
        for layer in baseline.model.layers:
            if layer.block_type == "linear_attention":
                layer.linear_attn.chunk_gated_delta_rule = partial(model_kernel, cut=layout.offsets[1])
    packing = TreePacking(gdn_conv_kernel_size=4)
    training_row = packing.pack(
        [
            dict(
                input_ids=row,
                group_id=0,
                completion_mask=[0] + [1] * (len(row) - 1),
                old_log_probs=[0.0] * len(row),
                advantage=1.0,
            )
            for row in rows
        ]
    )
    inputs = {name: tensor.cuda()[None] for name, tensor in vars(training_row).items()}
    mask = torch.ones_like(inputs["input_ids"], dtype=torch.bool)
    forward_kwargs = packing.forward_kwargs(inputs, mask)
    if checkpointing:
        forward_kwargs.update(fused_lm_head=True, shift_labels=inputs["shift_labels"])
    output = model(
        torch.tensor([packed], device="cuda"),
        position_ids=layout.position_ids.cuda()[None],
        **forward_kwargs,
    )
    baseline_ids = torch.zeros(len(rows), max(map(len, rows)), device="cuda", dtype=torch.long)
    baseline_mask = torch.zeros_like(baseline_ids)
    for r, row in enumerate(rows):
        baseline_ids[r, : len(row)] = torch.tensor(row, device="cuda")
        baseline_mask[r, : len(row)] = 1
    reference_output = baseline(baseline_ids, attention_mask=baseline_mask, use_cache=False).logits
    loss, ref_loss = 0, 0
    for r, row in enumerate(rows):
        index = [mapping[t] for t in forest.walk(row, 0)]
        ref = reference_output[r : r + 1, : len(row)]
        labels = torch.tensor(row[1:], device="cuda")
        ref_log_probs = ref[:, :-1].float().log_softmax(-1).squeeze(0).gather(1, labels[:, None]).squeeze(1)
        if checkpointing:
            torch.testing.assert_close(output.log_probs[0, index[:-1]], ref_log_probs, atol=0.025, rtol=0.004)
        else:
            torch.testing.assert_close(output.logits[:, index], ref, atol=0.025, rtol=0.04)
            loss = loss + F.cross_entropy(output.logits[:, index[:-1]].float().squeeze(0), labels, reduction="sum")
        ref_loss = ref_loss + F.cross_entropy(ref[:, :-1].float().squeeze(0), labels, reduction="sum")
    if checkpointing:
        loss = -output.log_probs.flatten()[training_row.token_index.cuda()].sum()
    torch.testing.assert_close(loss, ref_loss, atol=0.05, rtol=0.002)
    loss.backward()
    ref_loss.backward()
    max_error = 0
    for (name, actual), (_, expected) in zip(model.named_parameters(), baseline.named_parameters(), strict=True):
        assert actual.grad is not None, name
        torch.testing.assert_close(actual.grad, expected.grad, atol=0.15, rtol=0.08, msg=name)
        relative_rms = (actual.grad.float() - expected.grad.float()).norm() / expected.grad.float().norm().clamp_min(
            1e-8
        )
        assert relative_rms < 0.04, (name, relative_rms.item(), actual.grad.norm().item(), expected.grad.norm().item())
        max_error = max(max_error, relative_rms.item())
    record_property("max_gradient_relative_rms", max_error)
