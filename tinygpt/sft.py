"""Supervised fine-tuning on chat-formatted data."""
import torch
from torch.cuda.amp import GradScaler, autocast
from config import TinyGPTConfig
from model import TinyGPT
from dataset import get_sft_loader
from utils import setup_logger, save_checkpoint, log_metrics, load_checkpoint

def sft(cfg, checkpoint_path=None):
    logger = setup_logger("sft", cfg.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TinyGPT(cfg).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=cfg.sft_lr, weight_decay=cfg.weight_decay)
    if checkpoint_path: load_checkpoint(model, opt, checkpoint_path)
    scaler = GradScaler()
    loader = get_sft_loader(cfg)
    for epoch in range(cfg.sft_epochs):
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with autocast(): _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
            if step % 50 == 0: log_metrics(logger, epoch*len(loader)+step, loss.item(), cfg.sft_lr)
        save_checkpoint(model, opt, epoch, cfg, tag=f"sft_epoch{epoch}")

if __name__ == "__main__":
    import sys; cfg = TinyGPTConfig()
    sft(cfg, sys.argv[1] if len(sys.argv)>1 else None)
