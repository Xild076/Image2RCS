import torch

model = torch.nn.Linear(10, 10).to("mps")
x = torch.randn(1, 10).to("mps")
cmodel = torch.compile(model)
try:
    cmodel(x)
    print("Compile worked!")
except Exception as e:
    print("Compile failed:", e)
