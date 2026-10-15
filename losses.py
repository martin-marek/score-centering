"""REINFORCE with importance-sampling corrections and score centering.

Every correction is REINFORCE with a detached weight w(r) on the ratio
r = p/q of trainer (p) to sampler (q) probabilities, per token or per
sequence. Score centering subtracts the expected score under the effective
sampling mass q*w, which cancels the drift that a train/inference mismatch
adds to the gradient; it composes with any token-level ratio correction.
"""
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

sg = jax.lax.stop_gradient

# advantage of a (solved, unsolved) rollout; group_centered also subtracts the group mean
REWARD_MODES = {"plus_minus": (1.0, -1.0), "plus_zero": (1.0, 0.0),
                "zero_minus": (0.0, -1.0), "group_centered": (1.0, -1.0)}


def compute_advantages(rewards, group_size, reward_mode):
    solved, unsolved = REWARD_MODES[reward_mode]
    advantages = np.where(np.asarray(rewards) > 0, solved, unsolved).astype(np.float32)
    if reward_mode == "group_centered":
        groups = advantages.reshape(-1, group_size)
        advantages = (groups - groups.mean(1, keepdims=True)).reshape(-1)
    return advantages


DUAL_CLIP = 3.0  # PPO's ratio bound for negative advantages (arXiv:1912.09729)


class ISCorrection(NamedTuple):
    """One point on the IS-correction grid. TIS, MIS, PPO, DAPO, GSPO, TOPR and
    DPPO are all spellings of it (table in README.md)."""
    level: str = "token"   # token | seq_mean (mean log r) | seq_sum (sum log r: trajectory ratio)
    stat: str = "ratio"    # band tests the ratio r, or the TV distance |q - p| (DPPO; high = delta)
    low: float = 0.0       # band [low, high] on r; 0 / inf leave a side open
    high: float = math.inf
    outside: str = "drop"  # clamp: w = nearest bound | drop: w = 0
    sign: str = "both"     # which advantages get corrected: both | neg | pos (others w = 1)
                           # | ppo: block only moves away from q; A>0: (0, high], A<0: [low, DUAL_CLIP]

    def check(self):
        choices = dict(level=("token", "seq_mean", "seq_sum"), stat=("ratio", "tv"), outside=("clamp", "drop"),
                       sign=("both", "neg", "pos", "ppo"))
        for field, allowed in choices.items():
            assert getattr(self, field) in allowed, f"rl.is.{field} must be one of {allowed}"
        assert self.stat == "ratio" or self.level != "seq_sum", "seq_sum requires stat=ratio"
        return self


def log_mass(log_p, log_q, positive, c, tv=None):
    """log(q * w(p/q)): a token's effective sampling mass under correction c
    (-inf where w = 0, log q for no correction). Log space keeps the common
    in-band case q * (p/q) = p exact and overflow-free. The same function gives
      the IS weight w(r)       as the mass at p = r, q = 1,
      the centering mass q*w   as the mass at (p, q),
      the top-k tail factor    rho * w(1/rho) as the mass at p = 1, q = rho."""
    if c is None:
        return log_q
    lo = log_q + math.log(c.low) if c.low > 0 else -math.inf
    hi = log_q + math.log(c.high) if c.high < math.inf else math.inf
    if c.stat == "tv":
        in_band = tv <= c.high
        if c.sign == "ppo":
            in_band |= jnp.where(positive, log_p <= log_q, log_p >= log_q)
    elif c.sign == "ppo":
        in_band = jnp.where(positive, log_p <= hi, (log_p >= lo) & (log_p <= log_q + math.log(DUAL_CLIP)))
    else:
        in_band = (log_p >= lo) & (log_p <= hi)
    log_m = jnp.where(in_band, log_p, jnp.clip(log_p, lo, hi) if c.outside == "clamp" else -jnp.inf)
    if c.sign == "neg":
        log_m = jnp.where(positive, log_q, log_m)
    if c.sign == "pos":
        log_m = jnp.where(positive, log_m, log_q)
    return log_m


def is_weight(log_ratio, positive, c, tv=None):
    """Detached IS weight w(r), with r = exp(log_ratio) clipped to e^±20."""
    return sg(jnp.exp(log_mass(jnp.clip(log_ratio, -20.0, 20.0), 0.0, positive, c, tv)))


def take_vocab(logp, ids):
    """take_along_axis over a [B, C, V] vocab axis that may be TP-sharded (the
    gather drops that axis, so its output sharding must be explicit)."""
    spec = jax.typeof(logp).sharding.spec
    if spec[-1] is None:
        return jnp.take_along_axis(logp, ids, -1)
    b = jnp.arange(ids.shape[0])[:, None, None]
    c = jnp.arange(ids.shape[1])[None, :, None]
    return logp.at[b, c, ids].get(out_sharding=P(*spec[:-1], None))


def loss_fn_drift(hidden, head, batch, chunk=128):
    """Surrogate whose parameter gradient is the token-mean expected score.

    Uses the unweighted sampler's transported top-k head + rho*p tail, with
    the same tail clipping and denominator floor as loss_fn_rl. The detached
    residual uses sum_v p(v) * grad log p(v) = 0 to account for the tail.
    Differentiate through both hidden and head to measure parameter drift;
    take the norm only after averaging the score vectors over live tokens.
    """
    mask = jnp.roll(batch["mask"], -1, axis=1).at[:, -1].set(False)
    seq_len = mask.shape[1]
    chunk = next(n for n in range(min(chunk, seq_len), 0, -1) if seq_len % n == 0)

    def chunk_sum(start):
        slc = lambda x: jax.lax.dynamic_slice_in_dim(x, start, chunk, axis=1)
        logp = jax.nn.log_softmax(head(slc(hidden)).astype(jnp.float32))
        logp_head = take_vocab(logp, slc(batch["center_ids"]))
        p_head, q_head = jnp.exp(logp_head), jnp.exp(slc(batch["center_logprobs"]))
        rho = jnp.maximum(1.0 - q_head.sum(-1), 0.0) / jnp.maximum(1.0 - p_head.sum(-1), 1e-6)
        surrogate = (sg(q_head - rho[..., None] * p_head) * logp_head).sum(-1)
        return (surrogate * slc(mask)).sum()

    chunks = jax.lax.map(jax.checkpoint(chunk_sum), jnp.arange(0, seq_len, chunk))
    return chunks.sum() / jnp.maximum(mask.sum(), 1)


def loss_fn_rl(hidden, head, batch, correction=None, score_center=False, chunk=128):
    """REINFORCE loss on sampled rollouts, with an optional ISCorrection and
    score centering. Returns (loss, metrics).

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B],
    sampling_token_logprobs, and for score centering either the sampler's full
    logprobs sampling_vocab_logprobs [B, T, V] (exact) or its top-k head
    center_ids / center_logprobs [B, T, k]. With the head, q's tail is modeled
    as rho * p (rho matches the tail mass). Every ratio correction is constant
    on that tail, so its centering term is alpha * E_p[log p], whose gradient is
    zero: the backward pass only touches the k gathered logprobs.
    """
    c = correction
    token_c = c if c and c.level == "token" else None
    seq_c = c if c and c.level != "token" else None
    full = "sampling_vocab_logprobs" in batch
    if score_center:
        assert full or "center_ids" in batch, "score centering needs sampler vocab logprobs"
        assert not c or c.stat == "ratio", "score centering composes only with stat=ratio"
    targets = jnp.roll(batch["tokens"], -1, axis=1)
    mask = jnp.roll(batch["mask"], -1, axis=1).at[:, -1].set(False)
    positive = batch["advantages"][:, None] >= 0
    seq_len = targets.shape[1]
    chunk = next(n for n in range(min(chunk, seq_len), 0, -1) if seq_len % n == 0)
    sampler_keys = ("sampling_token_logprobs", "sampling_vocab_logprobs", "center_ids", "center_logprobs")
    per_token = {"hidden": hidden, "targets": targets, "mask": mask} | {k: batch[k] for k in sampler_keys if k in batch}

    def chunk_sums(start):
        b = {k: jax.lax.dynamic_slice_in_dim(v, start, chunk, axis=1) for k, v in per_token.items()}
        logp_vocab = jax.nn.log_softmax(head(b["hidden"]).astype(jnp.float32))
        logp = take_vocab(logp_vocab, b["targets"][..., None])[..., 0]
        logq = b["sampling_token_logprobs"]
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        w = is_weight(logp - logq, positive, token_c, tv)
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # E_p[log p] has zero gradient
        terms = {"score": w * logp, "entropy": -plogp, "kl_samp_train": sg(logq - logp), "masked": w == 0, "tv": tv}
        if score_center and full:
            logq_vocab = b["sampling_vocab_logprobs"].astype(jnp.float32)
            mass = jnp.exp(log_mass(logp_vocab, logq_vocab, positive[..., None], token_c))
            terms["score"] -= (sg(mass) * logp_vocab).sum(-1)
        if "center_ids" in b:
            logp_head, logq_head = take_vocab(logp_vocab, b["center_ids"]), b["center_logprobs"]
            p_head, q_head = sg(jnp.exp(logp_head)), jnp.exp(logq_head)
            q_tail = 1.0 - q_head.sum(-1)
            rho = jnp.maximum(q_tail, 0.0) / jnp.maximum(1.0 - p_head.sum(-1), 1e-6)
            terms["kl_head"] = sg((q_head * (logq_head - logp_head)).sum(-1) + q_tail * jnp.log(jnp.maximum(rho, 1e-9)))
            if score_center:
                alpha = jnp.exp(log_mass(0.0, jnp.log(rho), positive, token_c))
                mass = jnp.exp(log_mass(logp_head, logq_head, positive[..., None], token_c))
                residual = mass - alpha[..., None] * p_head
                terms["score"] -= (sg(residual) * logp_head).sum(-1) + sg(alpha * plogp)
        return {k: (v * b["mask"]).sum(-1) for k, v in terms.items()}

    chunks = jax.lax.map(jax.checkpoint(chunk_sums), jnp.arange(0, seq_len, chunk))
    sums = {k: v.sum(0) for k, v in chunks.items()}  # per-sequence sums [B]

    n_tokens = jnp.maximum(mask.sum(), 1)
    advantages = batch["advantages"]
    frac_masked = sums["masked"].sum() / n_tokens
    if seq_c:
        seq_tokens = jnp.maximum(mask.sum(1), 1)
        log_ratio = -sums["kl_samp_train"]
        if seq_c.level == "seq_mean":
            log_ratio = log_ratio / seq_tokens
        w = is_weight(log_ratio, advantages >= 0, seq_c, sums["tv"] / seq_tokens)
        advantages = advantages * w
        frac_masked = (w == 0).mean()  # fraction of sequences
    loss = -(advantages * sums["score"]).sum() / n_tokens
    metrics = {k: sums[k].sum() / n_tokens for k in ("entropy", "kl_samp_train", "kl_head") if k in sums}
    metrics["pol_entropy"] = metrics.pop("entropy")
    return loss, metrics | {"frac_masked": frac_masked}
