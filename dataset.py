"""Dataset loading, tokenization, DataLoader creation."""
import os, json, torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from config import TinyGPTConfig

class PretrainDataset(Dataset):
    def __init__(self, data, ctx): self.data, self.ctx = data, ctx
    def __len__(self): return len(self.data) - self.ctx
    def __getitem__(self, i):
        c = self.data[i:i+self.ctx+1]; return c[:-1], c[1:]

class SFTDataset(Dataset):
    def __init__(self, path, sp, ctx):
        self.ctx, self.samples = ctx, []
        for line in open(path):
            ex = json.loads(line)
            text = f"<|system|>{ex.get('system','')}<|end|>"
            for t in ex.get("messages", []):
                text += f"<|{t['role']}|>{t['content']}<|end|>"
            ids = sp.encode(text, add_bos=True, add_eos=True)[:ctx+1]
            self.samples.append(torch.tensor(ids, dtype=torch.long))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): s=self.samples[i]; return s[:-1], s[1:]

def get_pretrain_loader(cfg):
    data = torch.load(os.path.join(cfg.data_dir, "train.bin"))
    return DataLoader(PretrainDataset(data, cfg.context_length), batch_size=cfg.batch_size, shuffle=True)

def get_sft_loader(cfg):
    sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
    return DataLoader(SFTDataset(os.path.join(cfg.sft_dir,"chat_examples.jsonl"), sp, cfg.context_length), batch_size=cfg.batch_size, shuffle=True)
