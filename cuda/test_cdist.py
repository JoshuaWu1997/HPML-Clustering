import torch
import my_cuda_util
import time

a = torch.rand(10000, 1000).cuda()
b = torch.rand(10000, 1000).cuda()
batch_size = a.shape[0] * a.shape[1] * b.shape[0] // 200000000
p = 5

torch.cuda.synchronize()
start = time.time()
for _ in range(1):
    y = my_cuda_util.mdist(a, b, p) ** (1 / p)
torch.cuda.synchronize()
print(time.time() - start)

torch.cuda.synchronize()
start = time.time()
for _ in range(1):
    x = [((a.unsqueeze(1) - bx.unsqueeze(0)).abs() ** p).sum(-1) for bx in b.chunk(batch_size)]
    x = torch.cat(x, dim=-1) ** (1 / p)

torch.cuda.synchronize()
print(time.time() - start)


print(x[:4, :4])
print(y[:4, :4])
print((x - y).abs().mean())
