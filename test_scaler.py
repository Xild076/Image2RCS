import torch
print(torch.__version__)
if torch.backends.mps.is_available():
    try:
        from torch.amp import GradScaler
        scaler = GradScaler("mps")
        print("MPS GradScaler worked")
    except Exception as e:
        print("Scaler Error:", e)
