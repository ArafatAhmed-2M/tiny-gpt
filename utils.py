"""Shared helpers: logging, checkpointing, timing."""
import os, time, logging, torch

def setup_logger(name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in [logging.FileHandler(f"{log_dir}/{name}.log"), logging.StreamHandler()]:
        h.setFormatter(fmt); logger.addHandler(h)
    return logger

def log_metrics(logger, step, loss, lr):
    logger.info(f"step={step:>7d}  loss={loss:.4f}  lr={lr:.2e}")

def save_checkpoint(model, opt, step, cfg, tag=None):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = f"{cfg.checkpoint_dir}/{tag or f'step{step}'}.pt"
    torch.save({"step":step,"model":model.state_dict(),"optimizer":opt.state_dict(),"config":cfg.__dict__}, path)
    print(f"Saved → {path}")

def load_checkpoint(model, opt, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if opt and "optimizer" in ckpt: opt.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0)

class Timer:
    def __enter__(self): self.t = time.perf_counter(); return self
    def __exit__(self, *_): print(f"Elapsed: {time.perf_counter()-self.t:.3f}s")
