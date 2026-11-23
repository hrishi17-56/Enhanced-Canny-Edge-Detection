"""Edge-map metrics.

precision  share of predicted edge pixels that lie within `tol` pixels of a true edge pixel.
recall     share of true edge pixels that have a predicted edge pixel within `tol` pixels.
f1         harmonic mean of the two.

Limits: this is a simplified tolerance-based match. It does not enforce a
one-to-one pairing between predicted and true pixels (the Berkeley benchmark
does), so thick edges can score a little higher than they would there. Results
depend on the tolerance and on how the ground truth was drawn. Without a
ground-truth map these numbers cannot be computed. In that case use
agreement_f1, which only measures how much two methods agree.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


def edge_prf(pred, gt, tol=2):
    p, g = pred > 0, gt > 0
    n_p, n_g = int(p.sum()), int(g.sum())
    if n_p == 0 and n_g == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if n_p == 0 or n_g == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    dist_to_gt = ndi.distance_transform_edt(~g)
    dist_to_pred = ndi.distance_transform_edt(~p)
    precision = float((dist_to_gt[p] <= tol).mean())
    recall = float((dist_to_pred[g] <= tol).mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def agreement_f1(a, b, tol=2):
    """F1 between two predicted maps. This is agreement, not accuracy."""
    return edge_prf(a, b, tol)["f1"]


def edge_density(edges):
    return float((edges > 0).mean())
