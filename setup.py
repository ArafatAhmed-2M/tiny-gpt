"""Environment setup and TPU/GPU initialization."""
import os, torch

def setup_device():
    if "COLAB_TPU_ADDR" in os.environ:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"TPU device: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    return device

if __name__ == "__main__":
    setup_device()
