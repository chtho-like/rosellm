import torch

# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor,
          N: int, d_model: int, h: int):
    dk = d_model // h 
    q = Q.view(N, h, dk).transpose(0, 1)
    k = K.view(N, h, dk).transpose(0, 1)
    v = V.view(N, h, dk).transpose(0, 1)
    o = torch.zeros_like(q)
    q_blk_size, k_blk_size = min(16, N), min(16, N) 
    qblks = torch.split(q, q_blk_size, dim=1)
    kblks = torch.split(k, k_blk_size, dim=1)
    vblks = torch.split(v, k_blk_size, dim=1)
    oblks = list(torch.split(o, q_blk_size, dim=1))
    l = torch.zeros(*q.shape[:-1], 1, device=q.device, dtype=q.dtype)
    m = torch.full((*q.shape[:-1], 1), -1e10, device=q.device, dtype=q.dtype)
    lblks = list(torch.split(l, q_blk_size, dim=1))
    mblks = list(torch.split(m, q_blk_size, dim=1))
    scale = dk ** -0.5
    for j in range(len(kblks)):
        kj, vj = kblks[j], vblks[j]
        for i in range(len(qblks)):
            qi, oi, li, mi = qblks[i], oblks[i], lblks[i], mblks[i]
            x = qi @ kj.transpose(-1, -2) * scale 
            mblk, _ = torch.max(x, dim=-1, keepdim=True)
            pij = (x - mblk).exp()
            lblk = pij.sum(dim=-1, keepdim=True)
            pijvj = pij @ vj
            mi_new = torch.maximum(mblk, mi)
            i2new = (mi - mi_new).exp()
            b2new = (mblk - mi_new).exp()
            li_new = li * i2new + lblk * b2new 
            oblks[i] = oi * li / li_new * i2new + pijvj * b2new / li_new
            lblks[i], mblks[i] = li_new, mi_new 
    o = torch.cat(oblks, dim=1).transpose(0, 1).reshape(N, d_model)
    output.copy_(o)
