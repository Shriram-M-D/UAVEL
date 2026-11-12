"""A4/A6: tiny ViT masked autoencoder with optional per-zone bottleneck adapters."""
import torch, torch.nn as nn


class Adapter(nn.Module):
    """Residual bottleneck adapter; zero-init up-projection so it starts as identity."""
    def __init__(self, dim, r=16):
        super().__init__()
        self.down, self.up = nn.Linear(dim, r), nn.Linear(r, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(torch.relu(self.down(x)))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, adapter=None):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.n2(x))
        return adapter(x) if adapter is not None else x


class MAE(nn.Module):
    def __init__(self, img=64, patch=8, dim=128, depth=4, heads=4, ddim=64, ddepth=2, mask_ratio=0.75):
        super().__init__()
        self.patch, self.mask_ratio, self.np = patch, mask_ratio, (img // patch) ** 2
        self.embed = nn.Linear(3 * patch * patch, dim)
        self.pos = nn.Parameter(torch.randn(1, self.np, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.d_embed = nn.Linear(dim, ddim)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, ddim))
        self.d_pos = nn.Parameter(torch.randn(1, self.np, ddim) * 0.02)
        self.d_blocks = nn.ModuleList(Block(ddim, 4) for _ in range(ddepth))
        self.d_norm = nn.LayerNorm(ddim)
        self.d_out = nn.Linear(ddim, 3 * patch * patch)
        self.adapters = nn.ModuleDict()  # zone_id -> one adapter per encoder block

    def add_adapter(self, zone, r=16):
        self.adapters[str(zone)] = nn.ModuleList(Adapter(self.embed.out_features, r) for _ in self.blocks)

    def patchify(self, x):
        p = self.patch
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // p, p, W // p, p)
        return x.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, p * p * C)

    def unpatchify(self, t):
        p = self.patch
        B = t.shape[0]
        g = int(self.np ** 0.5)
        t = t.reshape(B, g, g, p, p, 3).permute(0, 5, 1, 3, 2, 4)
        return t.reshape(B, 3, g * p, g * p)

    def encode(self, tokens, zone=None):
        ads = self.adapters[str(zone)] if zone is not None and str(zone) in self.adapters else None
        for i, b in enumerate(self.blocks):
            tokens = b(tokens, ads[i] if ads is not None else None)
        return self.norm(tokens)

    def embed_images(self, x, zone=None):
        """Representation for downstream probing (mean over all patches, no masking)."""
        return self.encode(self.embed(self.patchify(x)) + self.pos, zone).mean(1)

    def forward(self, x, zone=None):
        t = self.patchify(x)
        B, N, _ = t.shape
        keep = int(N * (1 - self.mask_ratio))
        idx = torch.rand(B, N).argsort(1)
        ids_keep, ids_restore = idx[:, :keep], idx.argsort(1)
        z = self.embed(t) + self.pos
        z = torch.gather(z, 1, ids_keep[..., None].expand(-1, -1, z.shape[-1]))
        z = self.d_embed(self.encode(z, zone))
        full = torch.cat([z, self.mask_tok.expand(B, N - keep, -1)], 1)
        full = torch.gather(full, 1, ids_restore[..., None].expand(-1, -1, full.shape[-1])) + self.d_pos
        for b in self.d_blocks:
            full = b(full)
        pred = self.d_out(self.d_norm(full))
        mask = torch.ones(B, N)
        mask.scatter_(1, ids_keep, 0)
        loss = (((pred - t) ** 2).mean(-1) * mask).sum() / mask.sum()
        return loss, pred, mask
