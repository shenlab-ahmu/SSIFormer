"""SSIFormer: shared multi-scale anchor encoder; 4-layer pre-LN Transformer with rotary position embeddings over the 1-kb
loop-domain tokens, masked mean + max pooling; symmetric pair readout [a + c, a * c, |a - c|, span, distance].
Module and parameter names are those of the stored weights."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorEnc(nn.Module):
    """per scale: 2 x Conv1d(k5) + GELU, attention pooling over the 20 positions; 3 scales -> Linear(192, 128)."""

    def __init__(self, cin):
        super().__init__()
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv1d(cin, 64, 5, padding=2), nn.GELU(), nn.Conv1d(64, 64, 5, padding=2), nn.GELU())
                                    for _ in range(3)])
        self.att = nn.ModuleList([nn.Conv1d(64, 1, 1) for _ in range(3)])
        self.proj = nn.Linear(192, 128)

    def forward(self, a):  # a: B x 3 x C x 20
        z = []
        for s in range(3):
            h = self.convs[s](a[:, s])
            w = torch.softmax(self.att[s](h), -1)
            z.append((h * w).sum(-1))
        return self.proj(torch.cat(z, -1))


class Head(nn.Module):
    """symmetric pair readout."""

    def __init__(self, span_dim):
        super().__init__()
        self.dist = nn.Linear(1, 16)
        self.mlp = nn.Sequential(nn.Linear(128 * 3 + span_dim + 16, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, a, c, span, logd):
        z = [a + c, a * c, (a - c).abs(), span, self.dist(logd)]
        return self.mlp(torch.cat(z, -1)).squeeze(-1)


def valid_mask(lens, L, device):
    return torch.arange(L, device=device)[None, :] < lens[:, None]


def masked_pool(h, m):
    """h (B, L, D), m (B, L) bool -> [mean, max] (B, 2D)."""
    mf = m[..., None].to(h.dtype)
    mean = (h * mf).sum(1) / mf.sum(1).clamp(min=1)
    mx = h.masked_fill(~m[..., None], -1e4).max(1).values
    return torch.cat([mean, mx], -1)


def rope_cache(L, dh, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=device).float() / dh))
    ang = torch.arange(L, device=device).float()[:, None] * inv[None, :]
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):  # x (B, H, L, dh)
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h].float(), x[..., h:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).to(x.dtype)


class TLayer(nn.Module):
    """pre-LN Transformer layer (d 128, 4 heads, FFN 256 ReLU, dropout 0.1) with RoPE and scaled dot-product attention."""

    def __init__(self, d=128, nh=4, ff=256, p=0.1):
        super().__init__()
        self.nh, self.dh, self.p = nh, d // nh, p
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(p), nn.Linear(ff, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, keep, cos, sin):  # keep (B, 1, 1, L) bool, True = attend
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.nh, self.dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.ff(self.ln2(x)))


class SSIFormer(nn.Module):
    def __init__(self, cin_anchor, c_in):
        super().__init__()
        self.enc = AnchorEnc(cin_anchor)
        self.inp = nn.Linear(c_in, 128)
        self.layers = nn.ModuleList([TLayer() for _ in range(4)])
        self.head = Head(256)

    def span(self, x, lens):
        B, L, _ = x.shape
        m = valid_mask(lens, L, x.device)
        h = self.inp(x)
        cos, sin = rope_cache(L, 32, x.device)
        keep = m[:, None, None, :]
        for lyr in self.layers:
            h = lyr(h, keep, cos, sin)
        return masked_pool(h, m)

    def forward(self, A, Cc, span, lens, logd):
        return self.head(self.enc(A), self.enc(Cc), self.span(span, lens), logd)


def build(family, cin_anchor, c_in):
    assert family == 'ssiformer', family
    return SSIFormer(cin_anchor, c_in)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
