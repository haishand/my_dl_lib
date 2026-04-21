import torch

x = torch.tensor([[1.0, 2.0], [2.0, 3.0]], requires_grad=True)
z = x.pow(2).sum()
z.backward()
g = x.grad
print(g)
