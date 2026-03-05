"""Pretraining loop with mixed precision + cosine LR."""
import math, torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from config import TinyGPTConfig
from model import TinyGPT
from dataset import get_pretrain_loader
from utils import setup_logger, save_checkpoint, log_metrics

def cosine_lr(step, warmup, total, min_ratio=0.1):
    if step < warmup: return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))

def pretrain(cfg):
    logger = setup_logger("pretrain", cfg.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TinyGPT(cfg).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched  = LambdaLR(opt, lambda s: cosine_lr(s, cfg.warmup_steps, cfg.max_steps))
    scaler = GradScaler()
    loader = get_pretrain_loader(cfg)
    step   = 0
    for _ in range(99999):
        for x, y in loader:
            if step >= cfg.max_steps: break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with autocast(): _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            if step % 100  == 0: log_metrics(logger, step, loss.item(), sched.get_last_lr()[0])
            if step % 5000 == 0: save_checkpoint(model, opt, step, cfg)
            step += 1
        if step >= cfg.max_steps: break
    save_checkpoint(model, opt, step, cfg, tag="final")

if __name__ == "__main__":
    pretrain(TinyGPTConfig())
