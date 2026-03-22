import torch
print(torch.__version__)
if torch.backends.mps.is_available():
    with torch.autocast(device_type="mps"):
        print("MPS autocast block worked")
