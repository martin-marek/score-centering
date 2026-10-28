"""CPU regression tests for the unweighted sampler's parameter-space drift."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import PartitionSpec as P

import losses


class TinyModel:
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=3)

    def __init__(self):
        self.weights = {
            "forward": jnp.array([[0.0, 0.0], [0.3, -0.2], [-0.4, 0.5], [0.1, 0.7]]),
            "head": jnp.array([[0.4, -0.3, 0.2, 0.6], [-0.2, 0.5, 0.7, -0.4]]),
        }

    def forward(self, tokens, weights, *, dtype, logits=False, live=None, **kwargs):
        assert dtype == jnp.bfloat16
        spec = jax.typeof(tokens).sharding.spec
        embedded = (weights["forward"].at[tokens].get(out_sharding=P(*spec, None)) if any(spec)
                    else weights["forward"][tokens])
        hidden = jnp.tanh(embedded)
        if live is not None:
            hidden = hidden * live[..., None]
        return self.head(hidden, weights, dtype=dtype) if logits else hidden

    def head(self, hidden, weights, *, dtype):
        assert dtype == jnp.bfloat16
        if any(jax.typeof(hidden).sharding.spec):
            return jnp.einsum("btd,dv->btv", hidden, weights["head"], out_sharding=P("data", None, "model"))
        return hidden @ weights["head"]


def make_batch():
    tokens = jnp.array([[1, 2, 3, 0, 0, 0], [2, 3, 1, 2, 3, 0]])
    return {
        "tokens": tokens,
        "mask": jnp.array([[False, False, True, False, False, False],
                           [False, True, True, True, True, False]]),
        "center_ids": jnp.broadcast_to(jnp.array([1, 3]), (*tokens.shape, 2)),
        "center_logprobs": jnp.log(jnp.broadcast_to(jnp.array([0.4, 0.35]), (*tokens.shape, 2))),
        "advantages": jnp.array([1.0, -0.5]),
        "sampling_token_logprobs": jnp.full(tokens.shape, -1.5),
    }


def logits(model, weights, batch):
    hidden = model.forward(batch["tokens"], weights, dtype=jnp.bfloat16,
                           live=batch["tokens"] != model.tokenizer.pad_token_id)
    return model.head(hidden, weights, dtype=jnp.bfloat16)


def surrogate(model, weights, batch, chunk=128):
    hidden = model.forward(batch["tokens"], weights, dtype=jnp.bfloat16,
                           live=batch["tokens"] != model.tokenizer.pad_token_id)
    return losses.loss_fn_drift(hidden, lambda h: model.head(h, weights, dtype=jnp.bfloat16), batch, chunk)


def fixed_sampler(model, weights, batch):
    """Independent full-vocabulary oracle: explicitly fill head and tail slots.

    Freeze the modeled sampler at the reference parameters. No residual or
    zero-expected-score identity is used in the oracle's differentiation.
    """
    raw = np.asarray(logits(model, weights, batch))
    p = np.exp(raw - raw.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    q = np.zeros_like(p)
    for b in range(p.shape[0]):
        for t in range(p.shape[1]):
            ids = np.asarray(batch["center_ids"][b, t])
            head = np.exp(np.asarray(batch["center_logprobs"][b, t]))
            tail = np.ones(p.shape[-1], bool)
            tail[ids] = False
            q[b, t, ids] = head
            q[b, t, tail] = p[b, t, tail] * max(1.0 - float(head.sum()), 0.0) / max(float(p[b, t, tail].sum()), 1e-6)
    return jnp.asarray(q)


def oracle_gradient(model, weights, batch):
    q = fixed_sampler(model, weights, batch)
    active = [(b, t) for b in range(batch["tokens"].shape[0])
              for t in range(batch["tokens"].shape[1] - 1) if batch["mask"][b, t + 1]]

    def objective(w):
        logp = jax.nn.log_softmax(logits(model, w, batch))
        # Direct sum over the complete, fixed modeled distribution.
        return sum((q[b, t] * logp[b, t]).sum() for b, t in active) / len(active)

    return jax.grad(objective)(weights)


def assert_tree_close(actual, expected, atol=2e-6):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, atol=atol, rtol=2e-5)


def test_parameter_gradient_matches_fixed_full_distribution():
    model, batch = TinyModel(), make_batch()
    np.testing.assert_allclose(fixed_sampler(model, model.weights, batch).sum(-1), 1.0, atol=1e-7)
    expected = oracle_gradient(model, model.weights, batch)
    actual = jax.grad(lambda w: surrogate(model, w, batch))(model.weights)
    assert_tree_close(actual, expected)
    for part in ("forward", "head"):
        assert float(jnp.linalg.norm(actual[part])) > 1e-3
    assert float(optax.global_norm(actual)) > 1e-3


def test_matched_sampler_has_zero_drift():
    model, batch = TinyModel(), make_batch()
    logp = jax.nn.log_softmax(logits(model, model.weights, batch))
    batch["center_logprobs"] = jnp.take_along_axis(logp, batch["center_ids"], -1)
    actual = jax.grad(lambda w: surrogate(model, w, batch))(model.weights)
    assert_tree_close(actual, oracle_gradient(model, model.weights, batch))
    assert float(optax.global_norm(actual)) < 1e-6


def test_sampler_coefficients_are_detached():
    model, batch = TinyModel(), make_batch()
    q_grad = jax.grad(lambda q: surrogate(model, model.weights, batch | {"center_logprobs": q}))(batch["center_logprobs"])
    np.testing.assert_array_equal(q_grad, 0.0)
    # The fixed-q oracle also checks that rho and the p-dependent residual
    # contribute no derivatives, rather than differentiating the surrogate's value.
    assert_tree_close(jax.grad(lambda w: surrogate(model, w, batch))(model.weights),
                      oracle_gradient(model, model.weights, batch))


def test_unweighted_drift_ignores_advantages_and_sampled_is_ratios():
    model, batch = TinyModel(), make_batch()
    changed = batch | {"advantages": jnp.array([-100.0, 27.0]),
                       "sampling_token_logprobs": jnp.full(batch["tokens"].shape, -20.0)}
    assert_tree_close(jax.grad(lambda w: surrogate(model, w, batch))(model.weights),
                      jax.grad(lambda w: surrogate(model, w, changed))(model.weights))


@pytest.mark.parametrize("chunk", [1, 2, 3, 4, 5, 128])
def test_chunk_value_and_parameter_gradient(chunk):
    model, batch = TinyModel(), make_batch()
    result = jax.value_and_grad(lambda w: surrogate(model, w, batch, chunk))(model.weights)
    reference = jax.value_and_grad(lambda w: surrogate(model, w, batch, 1))(model.weights)
    assert_tree_close(result, reference)


def test_mask_alignment_token_average_and_zero_mask():
    model, batch = TinyModel(), make_batch()
    expected = oracle_gradient(model, model.weights, batch)
    assert_tree_close(jax.grad(lambda w: surrogate(model, w, batch))(model.weights), expected)
    # The rows have 1 and 4 generated tokens: combine score vectors by token
    # count before taking the norm, rather than averaging per-token norms.
    row_grads = []
    for b in range(2):
        row = {k: v[b:b + 1] for k, v in batch.items()}
        row_grads.append(jax.grad(lambda w: surrogate(model, w, row))(model.weights))
    combined = jax.tree.map(lambda a, b: (a + 4 * b) / 5, *row_grads)
    assert_tree_close(expected, combined)
    assert not np.isclose(float(optax.global_norm(combined)),
                          float((optax.global_norm(row_grads[0]) + 4 * optax.global_norm(row_grads[1])) / 5),
                          rtol=1e-3)
    empty = batch | {"mask": jnp.zeros_like(batch["mask"])}
    value, grads = jax.value_and_grad(lambda w: surrogate(model, w, empty))(model.weights)
    assert float(value) == 0
    assert float(optax.global_norm(grads)) == 0
    # A true first-position mask cannot wrap round to the final position.
    first_only = empty | {"mask": empty["mask"].at[:, 0].set(True)}
    assert float(surrogate(model, model.weights, first_only)) == 0


@pytest.mark.parametrize("q_head", [[0.7, 0.2], [0.7, 0.3], [0.7, 0.3000001]])
def test_near_zero_tail_is_finite_and_uses_upstream_guard(q_head):
    hidden = jnp.array([[[20.0, 19.0, -10.0, -11.0], [0.0, 0.0, 0.0, 0.0]]])
    batch = {"mask": jnp.array([[False, True]]),
             "center_ids": jnp.array([[[0, 1], [0, 1]]]),
             "center_logprobs": jnp.log(jnp.array([[q_head, q_head]]))}
    value, grad = jax.value_and_grad(lambda h: losses.loss_fn_drift(h, lambda x: x, batch))(hidden)
    assert bool(jnp.isfinite(value)) and bool(jnp.isfinite(grad).all())
    # A full-vocab direct sum still agrees at the denominator floor; do not
    # renormalize the head or tail to force this guarded model to sum to one.
    p = np.asarray(jax.nn.softmax(hidden[0, 0]))
    q = np.array([*q_head, 0.0, 0.0], np.float32)
    q[2:] = p[2:] * max(1 - float(np.exp(np.asarray(batch["center_logprobs"][0, 0])).sum()), 0) / 1e-6
    oracle = jax.grad(lambda h: (jnp.asarray(q) * jax.nn.log_softmax(h[0, 0])).sum())(hidden)
    assert_tree_close(grad, oracle, atol=5e-3)
