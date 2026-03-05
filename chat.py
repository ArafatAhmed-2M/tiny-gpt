"""Interactive CLI chat loop."""
import torch, sentencepiece as spm
from config import TinyGPTConfig
from inference import load_model, generate, decode

def chat(cfg, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp     = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
    model  = load_model(cfg, ckpt_path, device)
    print("TinyGPT Chat — type \'quit\' to exit\n")
    history = []
    while True:
        user = input("You: ").strip()
        if user.lower() in ("quit","exit"): break
        history.append(("user", user))
        text = "<|system|>You are TinyGPT, a helpful assistant.<|end|>"
        for role, msg in history: text += f"<|{role}|>{msg}<|end|>"
        text += "<|assistant|>"
        prompt = torch.tensor(sp.encode(text), dtype=torch.long)
        out    = generate(model, prompt, cfg, device=device)
        reply  = decode(sp, out[len(prompt):]).split("<|end|>")[0].strip()
        print(f"TinyGPT: {reply}\n")
        history.append(("assistant", reply))

if __name__ == "__main__":
    import sys; cfg = TinyGPTConfig()
    chat(cfg, sys.argv[1] if len(sys.argv)>1 else "checkpoints/final.pt")
