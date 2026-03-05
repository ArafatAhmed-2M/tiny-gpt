"""JSON prompting + output parser/validator."""
import json, re, torch
import sentencepiece as spm
from config import TinyGPTConfig
from inference import load_model, generate, decode

def generate_json(cfg, ckpt_path, user_message, schema_hint=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp     = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
    model  = load_model(cfg, ckpt_path, device)
    system = "Respond with valid JSON only." + (f" Schema: {schema_hint}" if schema_hint else "")
    text   = f"<|system|>{system}<|end|><|user|>{user_message}<|end|><|assistant|>"
    prompt = torch.tensor(sp.encode(text), dtype=torch.long)
    out    = generate(model, prompt, cfg, device=device)
    raw    = decode(sp, out[len(prompt):]).split("<|end|>")[0]
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:    return json.loads(m.group()) if m else None
    except: return None
