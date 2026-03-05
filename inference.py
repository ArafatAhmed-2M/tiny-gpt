"""Generation utilities: greedy and top-p sampling."""
import torch, torch.nn.functional as F
import sentencepiece as spm
from config import TinyGPTConfig
from model import TinyGPT

def load_model(cfg, ckpt_path, device):
    model = TinyGPT(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    return model.eval()

@torch.no_grad()
def generate(model, prompt_ids, cfg, strategy="top_p", device="cpu"):
    ids = prompt_ids.unsqueeze(0).to(device)
    for _ in range(cfg.max_new_tokens):
        logits, _ = model(ids[:, -cfg.context_length:])
        logits = logits[:, -1, :] / cfg.temperature
        if strategy == "greedy":
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = F.softmax(logits, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            mask = (sp.cumsum(-1) - sp) > cfg.top_p
            sp[mask] = 0; sp /= sp.sum(-1, keepdim=True)
            nxt = si.gather(-1, torch.multinomial(sp, 1))
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == 3: break
    return ids[0].tolist()

def decode(sp, ids): return sp.decode(ids)
