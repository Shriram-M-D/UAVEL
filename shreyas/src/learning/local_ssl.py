"""A4: local masked-reconstruction training for ONE simulated UAV client (its own partition only)."""
import time, torch
from ..models.mae import MAE


def train_local(x, epochs=30, bs=64, lr=1e-3, seed=0, model=None, params=None, zone=None, log=print):
    torch.manual_seed(seed)
    model = model or MAE()
    params = list(params) if params is not None else list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
    hist, t0 = [], time.time()
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(x))
        tot = n = 0
        for i in range(0, len(x), bs):
            b = perm[i:i + bs]
            loss, _, _ = model(x[b], zone)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            n += len(b)
        hist.append(tot / n)
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"  epoch {ep+1}/{epochs} recon_loss={hist[-1]:.4f}")
    return model, dict(loss_history=hist, train_seconds=time.time() - t0,
                       trainable_params=sum(p.numel() for p in params))
