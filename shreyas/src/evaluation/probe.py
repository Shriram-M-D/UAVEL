"""A5: linear probe on frozen representations (held-out labelled subset)."""
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def embed(model, x, zone=None, bs=256):
    model.eval()
    return torch.cat([model.embed_images(x[i:i + bs], zone) for i in range(0, len(x), bs)]).numpy()


def linear_probe(model, x_tr, y_tr, x_te, y_te, zone=None):
    a, b = embed(model, x_tr, zone), embed(model, x_te, zone)
    sc = StandardScaler().fit(a)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(a), y_tr)
    return float(clf.score(sc.transform(b), y_te))
