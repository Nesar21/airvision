import torch

def main() -> None:
    print("PyTorch version:", torch.__version__)
    print("MPS available :", torch.backends.mps.is_available())
    print("MPS built-in  :", torch.backends.mps.is_built())

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.randn(2, 3, device=device)
        y = x @ torch.randn(3, 4, device=device)
        print("Sample matmul on MPS OK, shape:", y.shape)
    else:
        print("WARNING: MPS not available; using CPU fallback.")
        device = torch.device("cpu")
        x = torch.randn(2, 3, device=device)
        y = x @ torch.randn(3, 4, device=device)
        print("Sample matmul on CPU OK, shape:", y.shape)

if __name__ == "__main__":
    main()
