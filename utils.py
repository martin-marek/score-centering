import json
from pathlib import Path

import jax
import optax
import wandb
from omegaconf import OmegaConf


def parse_cfg(default_cfg, validate=None):
    """YAML defaults overridden by key=value arguments, validated before W&B."""
    cfg = OmegaConf.create(default_cfg)
    OmegaConf.set_struct(cfg, True)  # unknown keys are errors
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    if validate is not None:
        validate(cfg)
    if jax.process_index() == 0:
        wandb.init(project=cfg.log.project, mode=cfg.log.wandb_mode, name=cfg.log.run_name,
                   config=OmegaConf.to_container(cfg, resolve=True))
    return cfg


def make_optimizer(name, lr, b2=0.999, weight_decay=1e-4, warmup_steps=0, decay_steps=0):
    """SGD or AdamW. Linear warmup from 0, then a constant LR, or with decay_steps a cosine decay to 0 at that step."""
    assert name in ("sgd", "adamw"), name
    if decay_steps:
        lr = optax.warmup_cosine_decay_schedule(0.0, lr, warmup_steps, decay_steps)
    elif warmup_steps:
        lr = optax.linear_schedule(0.0, lr, warmup_steps)
    return optax.sgd(lr) if name == "sgd" else optax.adamw(lr, b2=b2, weight_decay=weight_decay)


def start_run(cfg):
    """Create outputs/<project>/<run name> and save the resolved config there."""
    name = cfg.log.run_name or (wandb.run.id if jax.process_index() == 0 else "run")
    run_dir = Path(cfg.log.dir).expanduser() / cfg.log.project / name
    if jax.process_index() == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        config = OmegaConf.to_container(cfg, resolve=True)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(f"run_dir={run_dir}")
    return run_dir


def log_metrics(step, metrics, run_dir):
    metrics = {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}
    if jax.process_index() == 0:
        print(f"step {step}", metrics)
        wandb.log(metrics, step=step)
        with open(run_dir / "logs.jsonl", "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
