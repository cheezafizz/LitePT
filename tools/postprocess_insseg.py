"""
Post-processing for query-based InsSeg (MQ-v1m1) predictions: split any predicted
mask that is spatially disconnected into separate per-component instances.

Nothing in the model (masked cross-attention, BCE/Dice loss) is geometry-aware —
a query is free to cover two physically far-apart point clusters if their learned
features look similar, and mask_threshold only gates confidence, not distance.
Real objects are (near-)always a single spatially contiguous blob, so any
predicted mask with >1 connected component (at a radius small vs. inter-object
spacing) is either two merged/confused objects or noise -- splitting it can only
help precision, and the biggest fragment keeps the original instance's identity.

  radius: connectivity radius in meters. Default 0.005 (5mm) matches the
  clustering radius already validated for this dataset (see
  litept-2of3-2mm-insseg-collapse memory: 5mm cured a mis-set 3cm radius that
  merged sub-mm-apart objects). Objects here sit 6-18cm apart, so 5mm is small
  enough to never bridge two distinct objects while still tolerating normal
  point-cloud sampling gaps within one object.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def split_disconnected_masks(coord, masks, scores, classes, radius=0.005,
                              min_points=10):
    """Split every mask into its spatially-connected components.

    coord: (N, 3) f32. masks: (M, N) bool. scores/classes: (M,).
    Returns (new_masks, new_scores, new_classes, stats) where stats is a dict
    with counts of masks split and fragments dropped for logging.
    """
    new_masks, new_scores, new_classes = [], [], []
    n_split = 0
    n_dropped = 0
    for m in range(masks.shape[0]):
        idx = np.where(masks[m])[0]
        if idx.size == 0:
            continue
        if idx.size < min_points:
            new_masks.append(masks[m])
            new_scores.append(scores[m])
            new_classes.append(classes[m])
            continue

        tree = cKDTree(coord[idx])
        pairs = tree.query_pairs(radius, output_type="ndarray")
        n = idx.size
        if pairs.size:
            row = np.concatenate([pairs[:, 0], pairs[:, 1]])
            col = np.concatenate([pairs[:, 1], pairs[:, 0]])
            graph = coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(n, n))
        else:
            graph = coo_matrix((n, n))
        n_comp, labels = connected_components(graph, directed=False)

        if n_comp == 1:
            new_masks.append(masks[m])
            new_scores.append(scores[m])
            new_classes.append(classes[m])
            continue

        n_split += 1
        comp_sizes = np.bincount(labels, minlength=n_comp)
        for c in range(n_comp):
            comp_idx = idx[labels == c]
            if comp_idx.size < min_points:
                n_dropped += 1
                continue
            mask = np.zeros(coord.shape[0], dtype=bool)
            mask[comp_idx] = True
            new_masks.append(mask)
            new_scores.append(scores[m])
            new_classes.append(classes[m])

    stats = dict(n_split=n_split, n_dropped=n_dropped,
                 n_in=int(masks.shape[0]), n_out=len(new_masks))
    if not new_masks:
        return (np.zeros((0, coord.shape[0]), dtype=bool),
                np.zeros(0, np.float32), np.zeros(0, np.int32), stats)
    return (np.stack(new_masks), np.array(new_scores, np.float32),
            np.array(new_classes, np.int32), stats)
