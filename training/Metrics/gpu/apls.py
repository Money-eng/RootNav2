# Metrics/gpu/graph_connectivity.py
from skan import summarize
import torch

from ..base import BaseMetric


def _to_cu(x: torch.Tensor):
    import cupy as cp
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))


def _ensure_2d_bin(t: torch.Tensor, thr: float):
    if t.ndim == 4 and t.shape[1] == 1:
        t = t[:, 0]
    elif t.ndim == 4 and t.shape[1] > 1:
        raise ValueError("Mono-classe requis (C=1).")
    return (t >= thr).to(torch.uint8)


def _skeletonize_gpu(bin_cp):
    try:
        from cucim.skimage.morphology import thin
        return thin(bin_cp)
    except Exception:
        import numpy as np, cupy as cp
        from skimage.morphology import skeletonize
        sk = [skeletonize(cp.asnumpy(bin_cp[i]).astype(bool)) for i in range(bin_cp.shape[0])]
        return cp.asarray(np.stack(sk, axis=0))


def _neighbors_count(skel_cp):
    from cupyx.scipy.ndimage import convolve
    import cupy as cp
    k = cp.ones((3, 3), dtype=cp.int32);
    k[1, 1] = 0
    return convolve(skel_cp.astype(cp.int32), k, mode='constant', cval=0)


def _extract_graph(skel_np):
    """
    Construit un graphe 8-connexe (nodes: endpoints & jonctions; edges: chemins entre nodes).
    Retourne networkx.Graph avec positions 2D.
    """
    import numpy as np, networkx as nx
    H, W = skel_np.shape
    on = skel_np.astype(bool)
    # degré (voisinage 8)
    from scipy.ndimage import convolve
    k = np.ones((3, 3), np.int32);
    k[1, 1] = 0
    deg = convolve(on.astype(np.int32), k, mode='constant', cval=0)
    nodes = set(map(tuple, np.argwhere(on & ((deg == 1) | (deg >= 3)))))

    G = nx.Graph()
    for (y, x) in nodes:
        G.add_node((y, x), pos=(float(y), float(x)))

    # suivre les arêtes depuis chaque node en longeant les pixels de deg==2
    visited = set()

    def neighbors8(y, x):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0: continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx_ < W and on[ny, nx_]:
                    yield (ny, nx_)

    for src in list(nodes):
        for nxt in neighbors8(*src):
            if (src, nxt) in visited or (nxt, src) in visited:
                continue
            path = [src, nxt]
            cur = nxt
            prev = src
            while True:
                if cur in nodes and cur != src:
                    # fin d'arête
                    length = sum(
                        ((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2) ** 0.5 for i in
                        range(1, len(path)))
                    G.add_edge(src, cur, length=length)
                    for i in range(len(path) - 1):
                        visited.add((path[i], path[i + 1]))
                    break
                # avancer le long des deg==2
                nbrs = [p for p in neighbors8(*cur) if p != prev]
                if len(nbrs) == 0:
                    # cul-de-sac -> endpoint non listé au départ ?
                    if cur not in G:
                        G.add_node(cur, pos=(float(cur[0]), float(cur[1])))
                    length = sum(
                        ((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2) ** 0.5 for i in
                        range(1, len(path)))
                    G.add_edge(src, cur, length=length)
                    for i in range(len(path) - 1):
                        visited.add((path[i], path[i + 1]))
                    break
                elif len(nbrs) > 1:
                    # jonction interne: créer node
                    if cur not in G:
                        G.add_node(cur, pos=(float(cur[0]), float(cur[1])))
                    length = sum(
                        ((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2) ** 0.5 for i in
                        range(1, len(path)))
                    G.add_edge(src, cur, length=length)
                    for i in range(len(path) - 1):
                        visited.add((path[i], path[i + 1]))
                    break
                else:
                    path.append(nbrs[0]);
                    prev, cur = cur, nbrs[0]
    return G


def _match_nodes(G_gt, G_pr, tol=3.0):
    """
    Associe les noeuds par plus proche voisin (euclidien) sous tolérance.
    Retourne dict: node_gt -> node_pred (ou None).
    """
    import numpy as np
    from scipy.spatial import cKDTree
    gt_nodes = np.array([n for n in G_gt.nodes()], dtype=np.int32)
    pr_nodes = np.array([n for n in G_pr.nodes()], dtype=np.int32)
    if len(gt_nodes) == 0 or len(pr_nodes) == 0:
        return {}
    tree = cKDTree(pr_nodes)
    pairs = {}
    for n in gt_nodes:
        d, idx = tree.query(n, k=1)
        if d <= tol:
            pairs[tuple(n)] = tuple(pr_nodes[idx])
    return pairs


def _apls(G_gt, G_pr, pairs):
    """
    APLS approx: on échantillonne des paires de noeuds terminaux correspondants
    et compare les longueurs de plus courts chemins. [0..1], plus grand = meilleur.
    """
    import numpy as np, networkx as nx
    # terminaux (deg==1)
    terminals_gt = [n for n in G_gt.nodes() if G_gt.degree(n) == 1 and n in pairs]
    if len(terminals_gt) < 2:
        return 0.0
    # map inverse
    inv = {v: k for k, v in pairs.items()}
    terminals_pr = [pairs[n] for n in terminals_gt if pairs[n] in G_pr.nodes()]
    if len(terminals_pr) < 2:
        return 0.0

    # toutes paires (ou sous-échantillon si trop)
    pairs_idx = []
    for i in range(len(terminals_gt)):
        for j in range(i + 1, len(terminals_gt)):
            pairs_idx.append((i, j))
    if len(pairs_idx) > 2000:
        rng = np.random.default_rng(0)
        pairs_idx = rng.choice(pairs_idx, size=2000, replace=False).tolist()

    scores = []
    for i, j in pairs_idx:
        s_gt, t_gt = terminals_gt[i], terminals_gt[j]
        s_pr, t_pr = pairs[s_gt], pairs[t_gt]
        try:
            L_gt = nx.shortest_path_length(G_gt, s_gt, t_gt, weight='length')
        except Exception:
            continue
        try:
            L_pr = nx.shortest_path_length(G_pr, s_pr, t_pr, weight='length')
        except Exception:
            # pas de chemin en préd -> pire cas
            scores.append(0.0);
            continue
        if L_gt <= 1e-6:
            continue
        scores.append(max(0.0, 1.0 - abs(L_pr - L_gt) / L_gt))
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


class APLS(BaseMetric):
    type = "gpu"

    def __init__(self, threshold: float = 0.5, node_tol: float = 3.0):
        super().__init__()
        self.threshold = threshold
        self.node_tol = node_tol

    def is_better(self, old, new) -> bool:
        return new > old

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor):
        import networkx as nx
        pred = _ensure_2d_bin(prediction, self.threshold)
        gt = _ensure_2d_bin(mask, self.threshold)
        if not pred.is_cuda or not gt.is_cuda:
            raise ValueError("GraphConnectivity attend des tenseurs CUDA.")
        cp_pred = _to_cu(pred)
        cp_gt = _to_cu(gt)
        sk_pred = _skeletonize_gpu(cp_pred)
        sk_gt = _skeletonize_gpu(cp_gt)

        # batch -> moyenne (on agrège les graphes par image)
        topo_p, topo_r, apls_vals = [], [], [] # precision, recall, apls
        for i in range(sk_pred.shape[0]):
            skp = (sk_pred[i].get() > 0)  # CPU numpy
            skg = (sk_gt[i].get() > 0)
            Gp = _extract_graph(skp)
            Gg = _extract_graph(skg)
            pairs = _match_nodes(Gg, Gp, tol=self.node_tol)

            # recall: arêtes GT correctement connectées
            ok = 0
            tot = Gg.number_of_edges()
            for u, v, data in Gg.edges(data=True):
                if u in pairs and v in pairs:
                    u2, v2 = pairs[u], pairs[v]
                    try:
                        _ = nx.has_path(Gp, u2, v2)
                        if _:
                            ok += 1
                    except Exception:
                        pass
            topo_r.append(ok / tot if tot > 0 else 0.0)
            
            # precision: arêtes préd expliquées par GT
            okp = 0
            totp = Gp.number_of_edges()
            inv_pairs = {v: k for k, v in pairs.items()}
            for u, v, data in Gp.edges(data=True):
                if u in inv_pairs and v in inv_pairs:
                    u2, v2 = inv_pairs[u], inv_pairs[v]
                    try:
                        _ = nx.has_path(Gg, u2, v2)
                        if _:
                            okp += 1
                    except Exception:
                        pass
                    
            topo_p.append(okp / totp if totp > 0 else 0.0)
            apls_vals.append(_apls(Gg, Gp, pairs))
        return float(sum(apls_vals) / len(apls_vals))
import torch, numpy as np

def img(h=128,w=128):
    return torch.zeros((1,1,h,w), device='cuda')

def line(img, y, x0, x1):
    x = torch.arange(min(x0,x1), max(x0,x1)+1, device='cuda')
    img[0,0,y, x] = 1
    return img

# 1) Identique
gt = line(img(), 64, 16, 112)
pr = line(img(), 64, 16, 112)
metric = APLS(threshold=0.5, node_tol=3.0)
score = metric(pr, gt); assert abs(score-1.0) < 1e-3

# 2) Gap
pr2 = line(img(), 64, 16, 60); line(pr2, 64, 68, 112)
score2 = metric(pr2, gt); assert score2 < 0.5
print(f"APLS: {score=:.3f}, {score2=:.3f}")
    