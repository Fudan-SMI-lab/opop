import torch
a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
for _ in range(3):
    c = a @ b
torch.cuda.synchronize()
print("ran", c.shape)
