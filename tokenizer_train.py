"""Train SentencePiece tokenizer on raw text."""
import os, glob, sentencepiece as spm
from config import TinyGPTConfig

def train_tokenizer(cfg):
    raw = glob.glob("data/raw/*.txt") + glob.glob("data/raw/*.md")
    if not raw: raise FileNotFoundError("No files in data/raw/")
    with open("/tmp/corpus.txt", "w") as out:
        for fp in raw:
            out.write(open(fp).read() + "\n")
    os.makedirs("tokenizer", exist_ok=True)
    spm.SentencePieceTrainer.train(
        input="/tmp/corpus.txt", model_prefix="tokenizer/tinygpt",
        vocab_size=cfg.vocab_size, character_coverage=0.9995, model_type="bpe",
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols=["<|user|>","<|assistant|>","<|system|>","<|end|>"])
    print("Tokenizer saved.")

if __name__ == "__main__":
    train_tokenizer(TinyGPTConfig())
