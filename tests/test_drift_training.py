"""Exercise the real training loop with a tiny model and local fake rollouts."""
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P
from omegaconf import OmegaConf

import train_rl
from test_drift import TinyModel, assert_tree_close, make_batch, oracle_gradient


def config(every=0, vocab=2):
    cfg = OmegaConf.create(train_rl.DEFAULT_CFG)
    cfg.rl.log_drift_every = every
    cfg.sampler.vocab_logprobs = vocab
    return cfg


def forbid(*args, **kwargs):
    pytest.fail("unexpected model, sampler, external logger, or drift call")


@pytest.mark.parametrize("every", [-1, 0.0, 1.5, True, False, None, "", "bad", "2"])
def test_invalid_interval_fails_before_loading_or_logging(monkeypatch, every):
    monkeypatch.setattr(train_rl.models, "load", forbid)
    monkeypatch.setattr(train_rl.sampling, "generate", forbid)
    monkeypatch.setattr(train_rl.utils, "start_run", forbid)
    monkeypatch.setattr(train_rl.wandb, "init", forbid)
    with pytest.raises(ValueError, match="rl.log_drift_every must be 0 .*positive integer"):
        train_rl.train(config(every))


@pytest.mark.parametrize("vocab", [None, "full", True, False, 0, -2, 1.5, "invalid", "2"])
def test_invalid_head_fails_early(monkeypatch, vocab):
    monkeypatch.setattr(train_rl.models, "load", forbid)
    monkeypatch.setattr(train_rl.sampling, "generate", forbid)
    monkeypatch.setattr(train_rl.utils, "start_run", forbid)
    with pytest.raises(ValueError, match="positive integer top-k sampler.vocab_logprobs"):
        train_rl.train(config(1, vocab))


@pytest.mark.parametrize("override", ["rl.log_drift_every=-1", "rl.log_drift_every=0.5",
                                      "rl.log_drift_every=true", "rl.log_drift_every=bad",
                                      "rl.log_drift_every=1 sampler.vocab_logprobs=full",
                                      "rl.log_drift_every=1 sampler.vocab_logprobs=null"])
def test_cli_validation_precedes_wandb_init(monkeypatch, override):
    monkeypatch.setattr("sys.argv", ["train_rl.py", *override.split()])
    monkeypatch.setattr(train_rl.wandb, "init", forbid)
    with pytest.raises(ValueError, match="rl.log_drift_every"):
        train_rl.utils.parse_cfg(train_rl.DEFAULT_CFG, validate=train_rl.check_drift_config)


def test_valid_cli_and_disabled_head_validation(monkeypatch):
    calls = []
    monkeypatch.setattr("sys.argv", ["train_rl.py", "rl.log_drift_every=2", "sampler.vocab_logprobs=3"])
    monkeypatch.setattr(train_rl.wandb, "init", lambda **kwargs: calls.append(kwargs))
    cfg = train_rl.utils.parse_cfg(train_rl.DEFAULT_CFG, validate=train_rl.check_drift_config)
    assert cfg.rl.log_drift_every == 2 and len(calls) == 1
    assert OmegaConf.create(train_rl.DEFAULT_CFG).rl.log_drift_every == 0
    for vocab in (None, "full", 2):
        train_rl.check_drift_config(config(0, vocab))


class Dataset:
    def __len__(self):
        return 3

    def __getitem__(self, i):
        return {"prompt": [], "answer": "", "index": i}

    def shuffle(self, **kwargs):
        return self

    def select(self, indices):
        return self


def snapshot(tree):
    return jax.tree.map(lambda x: np.asarray(x).copy(), tree)


def run_tiny_training(monkeypatch, every):
    model, batch = TinyModel(), make_batch()
    model.init_kv = forbid  # sampling is replaced by the deterministic rollout below
    mesh = jax.make_mesh((1, 1), ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    model.weights = jax.tree.map(lambda w: jax.device_put(w, NamedSharding(mesh, P())), model.weights)
    cfg = config(every)
    cfg.rl.num_prompts, cfg.rl.group_size, cfg.rl.seq_len = 1, 2, 6
    cfg.rl.sc = False  # observation works independently of score centering
    cfg.rl.reward_mode = "plus_zero"
    cfg.eval.num_prompts, cfg.stop.steps, cfg.stop.collapse = 1, 5, 0
    cfg.dtypes.sampler, cfg.opt.optimizer = "float32", "adamw"
    cfg.log.wandb_mode = "disabled"
    records, keys, events, states, gradients, updates = [], [], [], [], [], []
    real_jit = jax.jit
    make_optimizer = train_rl.utils.make_optimizer

    def recording_optimizer(*args, **kwargs):
        optimizer = make_optimizer(*args, **kwargs)

        def update(grads, state, weights):
            delta, state = optimizer.update(grads, state, weights)
            jax.debug.callback(lambda g, u: (gradients.append(snapshot(g)), updates.append(snapshot(u))),
                               grads, delta, ordered=True)
            return delta, state

        return train_rl.optax.GradientTransformation(optimizer.init, update)

    def recording_jit(fun=None, **kwargs):
        if fun is None:
            return lambda f: recording_jit(f, **kwargs)
        compiled = real_jit(fun, **kwargs)
        kind = ("drift" if isinstance(fun, partial) and fun.func is train_rl.drift_norm
                else "optimize" if getattr(fun, "__name__", None) == "optimize" else None)
        if kind is None:
            return compiled

        def call(*args):
            events.append((kind, snapshot(args[0])))
            result = compiled(*args)
            if kind == "optimize":
                states.append(snapshot(result[1]))
            return result

        return call

    def generate(key, *args, **kwargs):
        keys.append(np.asarray(jax.random.key_data(key)).copy())
        put = lambda x, spec: jax.device_put(np.asarray(x), NamedSharding(mesh, spec))
        return (put(batch["tokens"], P("data", None)),
                put(batch["sampling_token_logprobs"], P("data", None)), None,
                put(batch["center_logprobs"], P("data", None, None)),
                put(batch["center_ids"], P("data", None, None)))

    with monkeypatch.context() as m:
        m.setattr(train_rl.models, "load", lambda *args: model)
        m.setattr(train_rl.rollout, "load_env", lambda *args, **kwargs: SimpleNamespace(get_dataset=Dataset, rubric=None))
        m.setattr(train_rl.rollout, "prompt_buffer", lambda *args: (np.asarray(batch["tokens"]), np.array([2, 1])))
        m.setattr(train_rl.rollout, "completions", lambda *args: (np.asarray(batch["mask"]), np.array([1, 4]), ["a", "b"]))
        m.setattr(train_rl.rollout, "score", lambda *args: np.array([1.0, 0.0], np.float32))
        m.setattr(train_rl.sampling, "generate", generate)
        m.setattr(train_rl.utils, "start_run", lambda cfg: None)
        m.setattr(train_rl.utils, "make_optimizer", recording_optimizer)
        m.setattr(train_rl.utils, "log_metrics", lambda step, metrics, run_dir: records.append((step, snapshot(metrics))))
        m.setattr(train_rl.wandb, "init", forbid)
        m.setattr(train_rl.wandb, "finish", lambda: None)
        m.setattr(train_rl.jax, "jit", recording_jit)
        if every == 0:
            m.setattr(train_rl, "drift_norm", forbid)
            m.setattr(train_rl.losses, "loss_fn_drift", forbid)
        with jax.set_mesh(mesh):  # models.load normally installs the trainer mesh
            train_rl.train(cfg)
            jax.effects_barrier()
    return model, batch, records, keys, events, states, gradients, updates


@pytest.mark.parametrize("every", [1, 3])
def test_observation_preserves_updates_rng_and_logs(monkeypatch, every):
    baseline = run_tiny_training(monkeypatch, 0)
    observed = run_tiny_training(monkeypatch, every)
    base_model, _, base_records, base_keys, base_events, base_states, base_grads, base_updates = baseline
    model, batch, records, keys, events, states, grads, updates = observed
    assert_tree_close(model.weights, base_model.weights, atol=0)
    for a, b in zip(jax.tree.leaves(model.weights), jax.tree.leaves(base_model.weights)):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(jax.tree.leaves(states), jax.tree.leaves(base_states)):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(jax.tree.leaves((grads, updates)), jax.tree.leaves((base_grads, base_updates))):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(keys, base_keys)
    assert [kind for kind, _ in base_events] == ["optimize"] * 5
    assert sum(kind == "drift" for kind, _ in events) == len(range(0, 5, every))
    event = 0
    for step, ((_, metrics), (_, base_metrics)) in enumerate(zip(records, base_records)):
        assert {"loss", "grad_norm", "pol_entropy", "kl_samp_train", "kl_head", "frac_masked",
                "reward", "frac_nonzero_advantage", "mean_completion_len", "frac_truncated"} <= base_metrics.keys()
        assert metrics.keys() - base_metrics.keys() == ({"drift_norm"} if step % every == 0 else set())
        assert ("eval_reward" in metrics) == (step in (0, 4))
        for name, value in base_metrics.items():
            np.testing.assert_array_equal(metrics[name], value)
        if step % every == 0:
            assert events[event][0] == "drift" and events[event + 1][0] == "optimize"
            pre_weights = events[event][1]
            assert_tree_close(pre_weights, events[event + 1][1], atol=0)
            # Check the actual logged norm through both parameter groups.
            expected = train_rl.optax.global_norm(oracle_gradient(model, pre_weights, batch))
            np.testing.assert_allclose(metrics["drift_norm"], expected, atol=2e-6, rtol=2e-5)
            event += 1
        assert events[event][0] == "optimize"
        assert_tree_close(events[event][1], base_events[step][1], atol=0)
        event += 1
    assert event == len(events)
