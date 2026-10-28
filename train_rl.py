"""Synchronous RL (REINFORCE + IS corrections + score centering) with a
deliberately mismatched sampler: stale, noisy, or quantized weights and KV cache."""
import os
from functools import partial

os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.85")
# No BFC preallocation: at 30B the fp32 masters pin the middle of the pool
# during sampling and fragment it below the trainer step's contiguous needs.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P, reshard
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import losses
import models
import rollout
import utils
from models import quant, sampling


DEFAULT_CFG = """
model:
  source: Qwen/Qwen3-1.7B
  weights_dir: ~/.cache/postax/weights
dtypes:
  sampler: bfloat16  # bfloat16 | int8 | float8_e4m3fn (8-bit weights and activations)
sampler:
  shard:
    tp: null  # sampler tensor parallelism (null: all devices)
  staleness: 0  # extra rollouts per sampler weight refresh
  weight_noise_scale: 0.0  # fixed relative Gaussian weight perturbation
  kv_quant_bits: 0  # fake-quantized KV-cache writes: 0 (off) | 2..16 (intN) | fp8 | fp4
  vocab_logprobs: 128  # sampler distribution kept for score centering: null | k (top-k) | full
rl:
  num_prompts: 16
  group_size: 8
  seq_len: 1024
  reward_mode: group_centered  # plus_minus | plus_zero | zero_minus | group_centered
  sc: false  # score centering
  log_drift_every: 0  # off; K>0: log ||E_q[grad log p]|| at step % K == 0 (including 0)
  is:  # IS correction; named methods are points on this grid (table in README.md)
    level: null    # null (off) | token | seq_mean | seq_sum
    stat: ratio    # ratio | tv
    low: 0.0
    high: .inf     # CLI overrides accept plain inf
    outside: drop  # clamp | drop
    sign: both     # both | neg | pos | ppo
env:
  id: countdown
  args: null  # keyword arguments for the environment's load_environment
  prompt_format: chat  # chat (post-trained models) | text (Base models)
opt:
  optimizer: sgd
  lr: 0.01
stop:
  steps: 100
  collapse: 0.5  # stop after 10 steps below this fraction of the peak reward (0 disables)
eval:
  num_prompts: 128  # held out from the training prompts; a multiple of rl.num_prompts
  every_steps: 50
log:
  project: postax-rl
  wandb_mode: online
  dir: ./outputs
  run_name: null
  save_model: false
run:
  seed: 0
"""


def fixed_weight_delta(weights, key, scale):
    """Relative Gaussian weight perturbation, drawn once per run."""
    leaves, structure = jax.tree.flatten(weights)
    keys = jax.random.split(key, len(leaves))
    deltas = [scale * jax.random.normal(k, w.shape, w.dtype, out_sharding=w.sharding) * w for k, w in zip(keys, leaves)]
    return jax.tree.unflatten(structure, deltas)


def check_drift_config(cfg):
    every = cfg.rl.log_drift_every
    if type(every) is not int or every < 0:
        raise ValueError("rl.log_drift_every must be 0 (off) or a positive integer")
    if every and (type(cfg.sampler.vocab_logprobs) is not int or cfg.sampler.vocab_logprobs <= 0):
        raise ValueError("rl.log_drift_every requires a positive integer top-k sampler.vocab_logprobs (not null or full)")


def drift_norm(model, weights, batch, chunk=128):
    """Norm of the mean unweighted sampler score, through forward and head."""
    def objective(weights):
        hidden = model.forward(batch["tokens"], weights, dtype=jnp.bfloat16, logits=False,
                               live=batch["tokens"] != model.tokenizer.pad_token_id)
        head = lambda x: model.head(x, weights, dtype=jnp.bfloat16)
        return losses.loss_fn_drift(hidden, head, batch, chunk)

    return optax.global_norm(jax.grad(objective)(weights))


def train(cfg):
    check_drift_config(cfg)
    key = jax.random.key(cfg.run.seed)
    model = models.load(cfg.model.source, cfg.model.weights_dir)
    model.weights = jax.tree.map(lambda w: w.astype(jnp.float32), model.weights)  # fp32 masters
    trainer_mesh = jax.tree.leaves(model.weights)[0].sharding.mesh
    optimizer = utils.make_optimizer(cfg.opt.optimizer, cfg.opt.lr)
    opt_state = optimizer.init(model.weights)
    run_dir = utils.start_run(cfg)
    is_cfg = {**cfg.rl["is"], "low": float(cfg.rl["is"].low), "high": float(cfg.rl["is"].high)}
    correction = None if is_cfg["level"] is None else losses.ISCorrection(**is_cfg).check()

    # Sampler: its own mesh, and weights that are a (possibly stale, noisy,
    # quantized) snapshot of the trainer's.
    sampler_tp = cfg.sampler.shard.tp or jax.device_count()
    sampler_mesh = jax.make_mesh((jax.device_count() // sampler_tp, sampler_tp), ("data", "model"),
                                 axis_types=(AxisType.Explicit, AxisType.Explicit))
    sampler_dtype = jnp.dtype(cfg.dtypes.sampler)
    quantized = sampler_dtype.itemsize == 1
    float_dtype = jnp.bfloat16 if quantized else sampler_dtype
    copy_to_sampler = jax.jit(
        lambda weights: {k: w.astype(float_dtype) if w.ndim > 1 and "norm" not in k else w for k, w in weights.items()},
        out_shardings={k: NamedSharding(sampler_mesh, w.sharding.spec) for k, w in model.weights.items()})
    scale = cfg.sampler.weight_noise_scale
    noise = copy_to_sampler(fixed_weight_delta(model.weights, jax.random.fold_in(key, 2), scale)) if scale else None

    @partial(jax.jit, donate_argnums=() if quantized else (0,))
    def perturb_and_quantize(weights, noise):
        if noise is not None:
            weights = jax.tree.map(jnp.add, weights, noise)
        return quant.quantize_tree(weights, sampler_dtype) if quantized else weights

    def sampler_snapshot(weights):
        weights = copy_to_sampler(jax.block_until_ready(weights))
        with jax.set_mesh(sampler_mesh):
            return perturb_and_quantize(weights, noise)

    sampler_forward = partial(model.forward, dtype=float_dtype, kv_quant_bits=cfg.sampler.kv_quant_bits)
    sampler_init_kv = partial(model.init_kv, dtype=float_dtype)
    # Eval samples in clean bf16: it measures the policy, not the sampler.
    eval_forward = partial(model.forward, dtype=jnp.bfloat16)

    env = rollout.load_env(cfg.env.id, **(OmegaConf.to_container(cfg.env.args) if cfg.env.args else {}))
    # Hub datasets are ordered by source: shuffle so they aren't a curriculum.
    dataset = env.get_dataset().shuffle(seed=cfg.run.seed)
    eval_rows = [dataset[i] for i in range(cfg.eval.num_prompts)]  # held out from training
    dataset = dataset.select(range(cfg.eval.num_prompts, len(dataset)))
    num_prompts, group_size, seq_len = cfg.rl.num_prompts, cfg.rl.group_size, cfg.rl.seq_len
    assert cfg.eval.num_prompts % num_prompts == 0, "eval.num_prompts must be a multiple of rl.num_prompts"
    token_sharding = NamedSharding(trainer_mesh, P("data", None))
    head_sharding = NamedSharding(trainer_mesh, P("data", None, None))

    def sample(rows, weights, key, forward, vocab_logprobs=None):
        """group_size scored completions per row. Returns the trainer-side batch,
        the full sampler logprobs if requested (still on the sampler mesh), the
        rewards, and rollout statistics."""
        rows = [{**row, "example_id": i} for i, row in enumerate(rows) for _ in range(group_size)]
        tokens, prompt_lengths = rollout.prompt_buffer(rows, model.tokenizer, seq_len, cfg.env.prompt_format)
        with jax.set_mesh(sampler_mesh):
            tokens, token_logprobs, vocab, head_logprobs, head_ids = sampling.generate(
                key, model, weights, forward, sampler_init_kv, tokens, prompt_lengths,
                logprobs_dtype=jnp.float32 if vocab_logprobs == "full" else None,
                logprobs_topk=vocab_logprobs if isinstance(vocab_logprobs, int) else 0)
            host_tokens = np.asarray(tokens)
        mask, lengths, texts = rollout.completions(host_tokens, prompt_lengths, model.tokenizer)
        batch = {"tokens": reshard(tokens, token_sharding), "mask": jax.device_put(mask, token_sharding),
                 "sampling_token_logprobs": reshard(token_logprobs, token_sharding)}
        if head_ids is not None:
            batch |= {"center_ids": reshard(head_ids, head_sharding),
                      "center_logprobs": reshard(head_logprobs, head_sharding)}
        truncated = (prompt_lengths + lengths >= seq_len) & (host_tokens[:, -1] != model.tokenizer.eos_token_id)
        stats = {"mean_completion_len": lengths.mean(), "frac_truncated": truncated.mean()}
        return batch, vocab, rollout.score(env.rubric, rows, texts, group_size), stats

    eval_key = jax.random.fold_in(key, 3)  # the same eval prompts and key every time

    def evaluate():
        weights = copy_to_sampler(jax.block_until_ready(model.weights))
        return np.mean([
            sample(eval_rows[i:i + num_prompts], weights, jax.random.fold_in(eval_key, i), eval_forward)[2]
            for i in range(0, len(eval_rows), num_prompts)])

    drift_norm_fn = jax.jit(partial(drift_norm, model)) if cfg.rl.log_drift_every else None

    @partial(jax.jit, donate_argnums=(0, 1))
    def optimize(weights, opt_state, batch):
        def objective(weights):
            # padding is dead weight in the MoE dispatch: mask it out
            hidden = model.forward(batch["tokens"], weights, dtype=jnp.bfloat16, logits=False,
                                   live=batch["tokens"] != model.tokenizer.pad_token_id)
            head = lambda x: model.head(x, weights, dtype=jnp.bfloat16)
            return losses.loss_fn_rl(hidden, head, batch, correction, cfg.rl.sc)

        (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(weights)
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)
        return weights, opt_state, {"loss": loss, "grad_norm": optax.global_norm(grads), **metrics}

    peak_reward = collapse_steps = 0
    for step in tqdm(range(cfg.stop.steps), disable=jax.process_index() != 0):
        if step % (cfg.sampler.staleness + 1) == 0:
            sampler_weights = sampler_snapshot(model.weights)
        metrics = {}
        if eval_rows and (step % cfg.eval.every_steps == 0 or step == cfg.stop.steps - 1):
            metrics["eval_reward"] = evaluate()

        rows = [dataset[(step * num_prompts + i) % len(dataset)] for i in range(num_prompts)]
        key, sample_key = jax.random.split(key)
        batch, vocab, rewards, stats = sample(
            rows, sampler_weights, sample_key, sampler_forward, cfg.sampler.vocab_logprobs)
        advantages = losses.compute_advantages(rewards, group_size, cfg.rl.reward_mode)
        batch["advantages"] = jax.device_put(advantages, NamedSharding(trainer_mesh, P("data")))
        if cfg.sampler.staleness == 0:
            del sampler_weights
        if vocab is not None:
            # Reshard only once the sampler weights are freed: source, destination
            # and collective scratch of this [B, T, V] array set the memory peak.
            batch["sampling_vocab_logprobs"] = reshard(vocab, NamedSharding(trainer_mesh, P("data", None, "model")))
            del vocab
        if drift_norm_fn is not None and step % cfg.rl.log_drift_every == 0:
            # Finish reading pre-update weights before optimize donates them.
            metrics["drift_norm"] = jax.block_until_ready(drift_norm_fn(model.weights, batch))
        model.weights, opt_state, opt_metrics = optimize(model.weights, opt_state, batch)
        del batch

        reward = rewards.mean()
        peak_reward = max(peak_reward, reward)
        collapse_steps = collapse_steps + 1 if reward < cfg.stop.collapse * peak_reward else 0
        utils.log_metrics(step, {"reward": reward, "frac_nonzero_advantage": (advantages != 0).mean(),
                                 **stats, **opt_metrics, **metrics}, run_dir)
        if collapse_steps == 10:
            print(f"reward collapsed below {cfg.stop.collapse} x peak: stopping")
            break

    if cfg.log.save_model and jax.process_index() == 0:
        models.save(model, run_dir / "final_model")
    if jax.process_index() == 0:
        wandb.finish()


if __name__ == "__main__":
    # Spawn, never fork: forked scorer workers would inherit this threaded CUDA process.
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    train(utils.parse_cfg(DEFAULT_CFG, validate=check_drift_config))
