# =============================================================================
# train_tokenizer.py — Train a SentencePiece BPE Tokenizer
# =============================================================================
# WHY SentencePiece?
# 1. No pre-tokenization needed — it works on raw text directly
# 2. Handles any language/character set without custom rules
# 3. Tiny dependency (single C++ library, no HuggingFace)
# 4. Same algorithm used by LLaMA, T5, and many production models
#
# WHY BPE specifically? (vs. Unigram)
# BPE builds vocabulary bottom-up by merging frequent character pairs.
# It's more predictable and produces slightly better results on English
# for small vocab sizes. Unigram is better for multilingual, which we
# don't need here.
#
# WHY 4096 vocab size?
# - Too small (<1000): many tokens per word → long sequences → slow training
# - Too large (>32000): sparse embeddings → hard to learn with small data
# - 4096 is a sweet spot for small corpora (10-100MB): most common English
#   words get their own token, rare words decompose into 2-3 subwords.
#   Powers of 2 are slightly more efficient for GPU/TPU matrix operations.
# =============================================================================

import sentencepiece as spm
import os


def train_tokenizer(
    corpus_path: str,
    model_dir: str,
    vocab_size: int = 4096,
    model_prefix: str = 'tinygpt',
    character_coverage: float = 0.9995,
):
    """
    Train a SentencePiece BPE tokenizer on the given corpus.
    
    Args:
        corpus_path: Path to the raw text file
        model_dir: Directory to save the trained tokenizer
        vocab_size: Number of tokens in the vocabulary
        model_prefix: Name prefix for saved files
        character_coverage: Fraction of characters to cover. 0.9995 for
                          English (drops extremely rare characters like
                          obscure Unicode). Use 1.0 for CJK languages.
    
    The trained tokenizer will have these special tokens:
        0: <pad>   — padding token (we'll use this for batch padding)
        1: <s>     — beginning of sequence
        2: </s>    — end of sequence
        3: <unk>   — unknown character (shouldn't appear with good coverage)
    
    WHY these specific special tokens?
    SentencePiece reserves IDs 0-2 by default. We explicitly configure them
    for clarity. The <pad> token is crucial — we need it for batching
    sequences of different lengths, and the model must learn to ignore it.
    """
    
    model_path = os.path.join(model_dir, model_prefix)
    
    # Check corpus exists and has content
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")
    
    file_size = os.path.getsize(corpus_path)
    print(f"📊 Corpus: {file_size / 1024 / 1024:.1f} MB")
    
    # Heuristic: if corpus is very small, reduce vocab size
    # Rule of thumb: you want at least ~50 examples per token for the
    # tokenizer to learn meaningful merges
    if file_size < 1024 * 1024:  # Less than 1MB
        suggested = min(vocab_size, 2048)
        if suggested < vocab_size:
            print(f"⚠️  Small corpus detected. Consider vocab_size={suggested}")
    
    print(f"🔧 Training SentencePiece BPE tokenizer (vocab_size={vocab_size})...")
    
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_path,
        vocab_size=vocab_size,
        model_type='bpe',                    # BPE algorithm
        character_coverage=character_coverage,
        
        # Special tokens configuration
        pad_id=0,                            # <pad> = 0
        bos_id=1,                            # <s> = 1 (begin of sequence)
        eos_id=2,                            # </s> = 2 (end of sequence)
        unk_id=3,                            # <unk> = 3
        
        # Training configuration
        num_threads=os.cpu_count(),          # Use all CPU cores
        shuffle_input_sentence=True,          # Randomize training order
        
        # Normalization — minimal, to preserve text structure
        # WHY minimal normalization? We want the model to see text roughly
        # as users will type it. Heavy normalization (lowercasing, etc.)
        # would make generation look unnatural.
        normalization_rule_name='identity',  # No normalization
        remove_extra_whitespaces=False,       # Keep whitespace as-is
        
        # Byte fallback — handle any character, even unseen ones
        # WHY? Without this, unseen characters become <unk>. With byte
        # fallback, they decompose into byte-level tokens. Costs a few
        # vocab slots but prevents information loss.
        byte_fallback=True,
        
        # Control vocabulary composition
        # These ensure common characters always get their own token
        split_digits=True,                    # Each digit is a separate token
        split_by_whitespace=True,             # Whitespace is a token boundary
        
        # Limit input sentence length for memory efficiency during training
        max_sentence_length=16384,
        input_sentence_size=1000000,          # Sample 1M sentences if corpus is huge
    )
    
    print(f"✅ Tokenizer saved to:")
    print(f"   Model: {model_path}.model")
    print(f"   Vocab: {model_path}.vocab")
    
    # Verify the tokenizer works
    verify_tokenizer(f"{model_path}.model")
    
    return f"{model_path}.model"


def verify_tokenizer(model_path: str):
    """
    Quick sanity check that the tokenizer encodes and decodes correctly.
    
    We test several important properties:
    1. Round-trip: encode → decode should recover original text
    2. Special tokens: BOS/EOS should be handled correctly
    3. Unknown handling: arbitrary characters shouldn't crash
    """
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    print(f"\n📋 Tokenizer Verification:")
    print(f"   Vocab size: {sp.get_piece_size()}")
    print(f"   PAD id: {sp.pad_id()}")
    print(f"   BOS id: {sp.bos_id()}")
    print(f"   EOS id: {sp.eos_id()}")
    print(f"   UNK id: {sp.unk_id()}")
    
    # Test encoding/decoding
    test_sentences = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "12345 + 67890 = 80235",
        "Let's test some edge cases: café, naïve, résumé.",
    ]
    
    print(f"\n   Sample encodings:")
    for sent in test_sentences:
        ids = sp.encode(sent)
        decoded = sp.decode(ids)
        pieces = sp.encode(sent, out_type=str)
        
        # Check round-trip fidelity
        match = "✅" if decoded == sent else "⚠️"
        print(f"   {match} \"{sent[:50]}\"")
        print(f"      → {len(ids)} tokens: {pieces[:10]}{'...' if len(pieces) > 10 else ''}")
        if decoded != sent:
            print(f"      ⚠️  Decoded: \"{decoded[:50]}\"")
    
    # Report average tokens per word (a useful metric)
    all_text = " ".join(test_sentences)
    tokens = sp.encode(all_text)
    words = all_text.split()
    ratio = len(tokens) / len(words)
    print(f"\n   Avg tokens/word: {ratio:.2f}")
    print(f"   (Good range: 1.2-2.0 for English)")


# ---------------------------------------------------------------------------
# Usage in Colab:
# ---------------------------------------------------------------------------
# corpus_path = os.path.join(BASE_DIR, 'data', 'raw', 'corpus.txt')
# model_dir = os.path.join(BASE_DIR, 'data', 'tokenizer')
# tokenizer_path = train_tokenizer(corpus_path, model_dir, vocab_size=4096)
# ---------------------------------------------------------------------------
