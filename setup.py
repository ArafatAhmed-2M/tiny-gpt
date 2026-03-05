# =============================================================================
# setup.py — Environment Setup & TPU Initialization
# =============================================================================
# This file handles three things:
# 1. Installing minimal dependencies
# 2. Detecting and initializing the TPU via torch_xla
# 3. Mounting Google Drive to access your dataset
#
# WHY torch_xla? Google Colab TPUs aren't CUDA devices. torch_xla bridges
# PyTorch to XLA (Accelerated Linear Algebra), which compiles and runs ops
# on TPU. The key mental model: your code looks like normal PyTorch, but
# tensors live on an XLA device instead of cuda/cpu, and you call
# xm.mark_step() to trigger actual computation (lazy evaluation).
# =============================================================================

import subprocess
import sys
import os

def install_dependencies():
    """
    Install only what we need. Minimalism is intentional:
    - torch + torch_xla: core framework + TPU bridge
    - sentencepiece: Google's tokenizer (no HuggingFace dependency)
    - No transformers, no datasets, no accelerate — we build everything.
    
    WHY these specific versions? Colab TPU runtimes ship with a specific
    torch_xla version. Mismatched versions = cryptic XLA errors.
    We let Colab's pre-installed torch_xla guide us and only add sentencepiece.
    """
    packages = [
        "sentencepiece",  # Tokenizer training and inference
    ]
    # torch and torch_xla should already be installed in Colab TPU runtime
    # If not, uncomment the following:
    # packages.extend(["torch", "torch_xla[tpu]"])
    
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    
    print("✅ Dependencies installed.")


def init_tpu():
    """
    Initialize TPU device via torch_xla.
    
    Returns the XLA device object you'll use everywhere instead of 'cuda:0'.
    
    WHY lazy tensors? torch_xla uses a trace-based execution model.
    Operations are recorded into a graph, and xm.mark_step() compiles
    and executes the graph on TPU. This means:
    - Don't call .item() or print tensor values mid-computation (breaks the graph)
    - Call xm.mark_step() at the end of each training step
    - Use xm.optimizer_step(optimizer) instead of optimizer.step()
    
    On a v5e-1 (single chip), we get one XLA device. No data parallelism
    needed — keeps our code simple.
    """
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        
        # Get the single TPU device
        device = xm.xla_device()
        
        print(f"✅ TPU initialized: {device}")
        print(f"   TPU type: {xm.xla_device_hw(str(device))}")
        
        return device
        
    except ImportError:
        print("⚠️  torch_xla not found. Falling back to CPU.")
        print("   To use TPU: Runtime → Change runtime type → TPU")
        import torch
        return torch.device('cpu')
    except Exception as e:
        print(f"⚠️  TPU init failed: {e}. Falling back to CPU.")
        import torch
        return torch.device('cpu')


def mount_drive():
    """
    Mount Google Drive so we can read datasets and save checkpoints.
    
    WHY Drive? Colab VMs are ephemeral — when the runtime disconnects,
    local files vanish. Drive persists across sessions. Put your raw
    text corpus in Drive before starting.
    """
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Create project directory structure in Drive
        base_dir = '/content/drive/MyDrive/tinygpt'
        dirs = [
            f'{base_dir}/data/raw',
            f'{base_dir}/data/tokenizer',
            f'{base_dir}/data/processed',
            f'{base_dir}/data/sft',
            f'{base_dir}/checkpoints',
            f'{base_dir}/logs',
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        
        print(f"✅ Drive mounted. Project dir: {base_dir}")
        return base_dir
        
    except ImportError:
        # Not running in Colab — use local directory
        base_dir = './tinygpt'
        dirs = [
            f'{base_dir}/data/raw',
            f'{base_dir}/data/tokenizer',
            f'{base_dir}/data/processed',
            f'{base_dir}/data/sft',
            f'{base_dir}/checkpoints',
            f'{base_dir}/logs',
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        
        print(f"✅ Using local directory: {base_dir}")
        return base_dir


def setup_all():
    """Run the complete setup sequence."""
    install_dependencies()
    base_dir = mount_drive()
    device = init_tpu()
    return base_dir, device


# ---------------------------------------------------------------------------
# Run this in your first Colab cell:
# ---------------------------------------------------------------------------
# from setup import setup_all
# BASE_DIR, DEVICE = setup_all()
# ---------------------------------------------------------------------------
