"""Transductive few-shot inference.

Methods:
  - SimpleShot      Wang et al. 2019
  - Mahalanobis     pooled covariance prototype classifier
  - LaplacianShot   Ziko et al. CVPR 2020
  - TIM             Boudiaf et al. NeurIPS 2020
  - Label Propagation Zhou et al. 2003
  - PT-MAP          Hu et al. ECCV 2022  (soft-EM + Sinkhorn, NEW)
  - alpha-TIM       Veilleux et al. NeurIPS 2021  (alpha-divergence TIM, NEW)
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F


# --------- preprocessing ---------

def tukey_transform(x: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """Tukey ladder of powers: sign(x) * |x|^beta (β=0.5 is default).
    Reduces feature distribution skew, often +1-3pt in few-shot
    (Yang et al. 'Free Lunch for Few-Shot Learning' 2021)."""
    if beta == 1.0:
        return x
    return np.sign(x) * np.power(np.abs(x), beta)


def center_and_normalize(feat_support: np.ndarray, feat_query: np.ndarray,
                         tukey_beta: float = 1.0,
                         use_combined_mean: bool = False):
    """Tukey (optional) → center → L2 normalize.

    use_combined_mean=True uses mean(support ∪ query) as the centering mean.
    This is transductive (uses test pixels, no labels) and is standard in
    few-shot literature (Wang 2019 'SimpleShot', Yang 2021 'Free Lunch')."""
    if tukey_beta != 1.0:
        feat_support = tukey_transform(feat_support, tukey_beta)
        feat_query = tukey_transform(feat_query, tukey_beta)
    if use_combined_mean:
        mu = np.concatenate([feat_support, feat_query], axis=0).mean(axis=0, keepdims=True)
    else:
        mu = feat_support.mean(axis=0, keepdims=True)
    s = feat_support - mu
    q = feat_query - mu
    s /= (np.linalg.norm(s, axis=1, keepdims=True) + 1e-12)
    q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    return s, q


def class_prototypes(feat_support: np.ndarray, labels_support: np.ndarray,
                     num_classes: int) -> np.ndarray:
    """Mean-of-class prototypes from support features (post-CN normalization).
    Returns [num_classes, D] L2-normalized prototypes."""
    D = feat_support.shape[1]
    P = np.zeros((num_classes, D), dtype=feat_support.dtype)
    for c in range(num_classes):
        m = (labels_support == c)
        if m.sum() == 0:
            continue
        P[c] = feat_support[m].mean(axis=0)
    P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    return P


# --------- SimpleShot ---------

def simpleshot(feat_support: np.ndarray, labels_support: np.ndarray,
               feat_query: np.ndarray, num_classes: int = 5,
               tukey_beta: float = 1.0,
               use_combined_mean: bool = False,
               return_probs: bool = False, temperature: float = 10.0):
    """Returns query predictions (int labels). Cosine NN to class prototypes.
    If return_probs, also returns softmax(τ·cos_sim) [Nq, C]."""
    s, q = center_and_normalize(feat_support, feat_query,
                                tukey_beta=tukey_beta,
                                use_combined_mean=use_combined_mean)
    P = class_prototypes(s, labels_support, num_classes)
    sim = q @ P.T
    preds = sim.argmax(axis=1)
    if return_probs:
        z = temperature * sim
        z = z - z.max(axis=1, keepdims=True)
        probs = np.exp(z); probs /= probs.sum(axis=1, keepdims=True)
        return preds, probs
    return preds


# --------- Mahalanobis prototype classifier ---------

def mahalanobis(feat_support: np.ndarray, labels_support: np.ndarray,
                feat_query: np.ndarray, num_classes: int = 5,
                tukey_beta: float = 1.0,
                use_combined_mean: bool = False,
                shrink: float = 0.3,
                return_probs: bool = False):
    """Gaussian discriminant with shared (pooled within-class) covariance.

    For each query q, predict argmin_c (q - μ_c)^T Σ^-1 (q - μ_c).
    Σ is estimated from support residuals after subtracting class means,
    then shrunk toward isotropic: Σ ← (1-shrink)·Σ + shrink·trace(Σ)/D·I.
    Useful when classes have different cluster shapes/anisotropy
    (Mahalanobis > cosine when it matters)."""
    s, q = center_and_normalize(feat_support, feat_query,
                                tukey_beta=tukey_beta,
                                use_combined_mean=use_combined_mean)
    D = s.shape[1]
    means = np.zeros((num_classes, D), dtype=s.dtype)
    residuals_list = []
    for c in range(num_classes):
        m = (labels_support == c)
        if m.sum() == 0:
            continue
        mu_c = s[m].mean(axis=0)
        means[c] = mu_c
        residuals_list.append(s[m] - mu_c)
    R = np.concatenate(residuals_list, axis=0)
    n_eff = max(1, R.shape[0] - num_classes)
    cov = (R.T @ R) / n_eff                                    # [D, D]
    # Ledoit-Wolf style shrinkage toward isotropic
    trace_avg = np.trace(cov) / D
    cov = (1.0 - shrink) * cov + shrink * trace_avg * np.eye(D, dtype=cov.dtype)
    # numerical floor
    cov += 1e-6 * np.eye(D, dtype=cov.dtype)
    cov_inv = np.linalg.inv(cov)
    # squared Mahalanobis distance to each class mean
    # d_c(x) = (x - μ_c)^T Σ^-1 (x - μ_c)
    # use the kernel trick: d = x^T Σ^-1 x - 2 x^T (Σ^-1 μ_c) + μ_c^T Σ^-1 μ_c
    W = means @ cov_inv                                        # [C, D]
    b = (means * W).sum(axis=1) * 0.5                          # [C]    (μ_c^T Σ^-1 μ_c)/2
    # score(c) ∝ x · (Σ^-1 μ_c) - (μ_c^T Σ^-1 μ_c)/2  — choose argmax
    scores = q @ W.T - b[None, :]
    preds = scores.argmax(axis=1)
    if return_probs:
        z = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(z); probs /= probs.sum(axis=1, keepdims=True)
        return preds, probs
    return preds


# --------- TIM: Transductive Information Maximization (Boudiaf NeurIPS 2020) ---------

def tim(feat_support: np.ndarray, labels_support: np.ndarray,
        feat_query: np.ndarray, num_classes: int = 5,
        tukey_beta: float = 1.0,
        use_combined_mean: bool = False,
        n_iter: int = 1000, lr: float = 1e-4,
        temperature: float = 15.0,
        lambda_marg: float = 1.0,
        lambda_cond: float = 0.1) -> np.ndarray:
    """TIM-GD inference.

    Loss minimized over class prototypes W:
        L = CE(support) - λ_marg · H(p̄_query)   +   λ_cond · E_query[H(p_i)]

      • CE forces W to classify support correctly
      • Maximizing H(p̄) encourages predictions to cover all classes
        (a natural prior against mode-collapse — exactly what we want for
        a large class-imbalanced test set)
      • Minimizing E[H(p_i)] encourages confident per-sample predictions

    All features are L2-normalized; classification uses cosine·temperature logits.
    """
    s, q = center_and_normalize(feat_support, feat_query,
                                tukey_beta=tukey_beta,
                                use_combined_mean=use_combined_mean)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = torch.tensor(s, dtype=torch.float32, device=device)
    Q = torch.tensor(q, dtype=torch.float32, device=device)
    y_S = torch.tensor(labels_support, dtype=torch.long, device=device)

    # init prototypes = L2-normalized class means
    W = torch.zeros(num_classes, S.shape[1], device=device)
    for c in range(num_classes):
        m = (y_S == c)
        if m.any():
            W[c] = S[m].mean(0)
    W = F.normalize(W, dim=1).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([W], lr=lr)

    for _ in range(n_iter):
        opt.zero_grad()
        Wn = F.normalize(W, dim=1)
        logits_s = temperature * (S @ Wn.T)
        logits_q = temperature * (Q @ Wn.T)
        loss_ce = F.cross_entropy(logits_s, y_S)
        prob_q = F.softmax(logits_q, dim=1)
        # marginal class distribution
        pbar = prob_q.mean(0).clamp_min(1e-12)
        h_marg = -(pbar * pbar.log()).sum()
        # conditional entropy (per-sample)
        h_cond = -(prob_q.clamp_min(1e-12) * prob_q.clamp_min(1e-12).log()).sum(1).mean()
        loss = loss_ce - lambda_marg * h_marg + lambda_cond * h_cond
        loss.backward()
        opt.step()

    with torch.no_grad():
        Wn = F.normalize(W, dim=1)
        preds = (Q @ Wn.T).argmax(1).cpu().numpy()
    return preds


# --------- Label Propagation (Zhou et al. 2003) ---------

def label_propagation(feat_support: np.ndarray, labels_support: np.ndarray,
                      feat_query: np.ndarray, num_classes: int = 5,
                      knn: int = 10, alpha: float = 0.7, sigma: float = 1.0,
                      tukey_beta: float = 1.0,
                      use_combined_mean: bool = False) -> np.ndarray:
    """Non-parametric transductive few-shot via label propagation.

    Pipeline:
      1. L2-normalize features (with optional Tukey + transductive centering)
      2. Build symmetric k-NN cosine-similarity graph over (support ∪ query)
      3. Symmetric-normalize:  S = D^{-1/2} W D^{-1/2}
      4. Closed-form propagation: F = (I − αS)^{-1} Y
         where Y[i, c]=1 for labeled support i in class c, 0 elsewhere
      5. Predict argmax over query rows of F

    Unlike SimpleShot / Mahalanobis (prototype-based) or LaplacianShot
    (prototype + smoothing), LP is fully *prototype-free* — labels can hop
    through query-query similarities (multi-step paths) to reach support
    seeds. Often helps when class clusters are elongated or non-convex.

    Args:
      knn:    k for the k-NN graph (default 10)
      alpha:  propagation strength (0=just original Y, 1=ignore seeds)
      sigma:  Gaussian kernel bandwidth on (1 - cos_sim)
    """
    s, q = center_and_normalize(feat_support, feat_query,
                                tukey_beta=tukey_beta,
                                use_combined_mean=use_combined_mean)
    X = np.concatenate([s, q], axis=0).astype(np.float32)
    n_s, n_q = s.shape[0], q.shape[0]
    n = n_s + n_q

    # k-NN cosine similarity (features already L2-normalized -> sim = dot)
    sim = X @ X.T
    np.fill_diagonal(sim, -np.inf)
    if knn >= n:
        knn = n - 1
    idx = np.argpartition(-sim, kth=knn, axis=1)[:, :knn]
    W = np.zeros((n, n), dtype=np.float32)
    rows = np.repeat(np.arange(n), knn)
    cols = idx.flatten()
    vals = sim[rows, cols]
    w = np.exp(-(1.0 - vals) / max(sigma, 1e-6))
    W[rows, cols] = w
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)

    # Symmetric normalization
    deg = W.sum(axis=1) + 1e-12
    D_inv_sqrt = 1.0 / np.sqrt(deg)
    S = (W * D_inv_sqrt[:, None]) * D_inv_sqrt[None, :]

    # Initial label matrix
    Y = np.zeros((n, num_classes), dtype=np.float32)
    Y[np.arange(n_s), labels_support.astype(np.int64)] = 1.0

    # Closed form: F = (I - alpha * S)^{-1} Y
    I = np.eye(n, dtype=np.float32)
    F_mat = np.linalg.solve(I - alpha * S, Y)

    preds = F_mat[n_s:].argmax(axis=1)
    return preds


# --------- LaplacianShot ---------

def _knn_affinity(q: np.ndarray, k: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Build a symmetric k-NN affinity matrix using cosine similarity.
    Returns dense [Nq, Nq] — only suitable for small query sets (Nq < ~500)."""
    Nq = q.shape[0]
    sim = q @ q.T                 # [Nq, Nq]
    np.fill_diagonal(sim, -np.inf)  # exclude self
    # take top-k per row
    if k >= Nq:
        k = Nq - 1
    idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    W = np.zeros((Nq, Nq), dtype=np.float32)
    rows = np.repeat(np.arange(Nq), k)
    cols = idx.flatten()
    vals = sim[rows, cols]
    # Gaussian kernel on cosine sim — convert to distance proxy
    w = np.exp(-(1.0 - vals) / max(sigma, 1e-6))
    W[rows, cols] = w
    # symmetrize: max(W, W.T)
    W = np.maximum(W, W.T)
    return W


def _knn_affinity_sparse(q: np.ndarray, k: int = 5, sigma: float = 1.0,
                          batch_size: int = 1000):
    """Build sparse symmetric k-NN affinity matrix for large query sets.

    Uses batch processing to avoid materializing the full Nq×Nq matrix.
    Memory: O(Nq * k) instead of O(Nq²).
    Returns scipy.sparse.csr_matrix.
    """
    import scipy.sparse as sp
    Nq = q.shape[0]
    actual_k = min(k, Nq - 1)

    rows_list, cols_list, vals_list = [], [], []
    q_f = q.astype(np.float32)       # save memory

    for start in range(0, Nq, batch_size):
        end = min(start + batch_size, Nq)
        # cosine sim block (features are L2-normalized)
        sim_block = q_f[start:end] @ q_f.T          # [batch, Nq]

        # mask self-similarities to -inf
        diag_idx = np.arange(start, end)
        sim_block[np.arange(end - start), diag_idx] = -np.inf

        # find top-k neighbors per row
        if actual_k < Nq - 1:
            idx = np.argpartition(-sim_block, kth=actual_k, axis=1)[:, :actual_k]
        else:
            idx = np.argsort(-sim_block, axis=1)[:, :actual_k]

        # gather similarity values
        gather_rows = np.arange(end - start)[:, None].repeat(actual_k, 1)
        batch_sims = sim_block[gather_rows, idx]

        # Gaussian kernel on cosine distance
        w = np.exp(-(1.0 - batch_sims) / max(sigma, 1e-6))

        rows_list.append(np.repeat(diag_idx, actual_k))
        cols_list.append(idx.flatten())
        vals_list.append(w.flatten())

        del sim_block                    # free per-block memory

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)

    # build sparse matrix and symmetrize (max)
    W = sp.coo_matrix((vals, (rows, cols)), shape=(Nq, Nq)).tocsr()
    W = W.maximum(W.T).tocsr()
    return W


def laplacianshot(feat_support: np.ndarray, labels_support: np.ndarray,
                  feat_query: np.ndarray, num_classes: int = 5,
                  knn: int = 5, lam: float = 1.0, n_iter: int = 20,
                  sigma: float = 1.0, tukey_beta: float = 1.0,
                  use_combined_mean: bool = False,
                  return_probs: bool = False):
    """LaplacianShot transductive inference.

    Args:
      knn:  k for the query affinity graph
      lam:  weight of the Laplacian smoothing term
      n_iter:  bound-optimization iterations
      sigma:  Gaussian kernel bandwidth on (1 - cosine)

    Returns int labels [Nq], or (labels, probs) if return_probs.

    Automatically uses sparse affinity matrix for Nq > 500 to handle
    large query sets (e.g. 61k test images) without OOM.
    """
    s, q = center_and_normalize(feat_support, feat_query,
                                tukey_beta=tukey_beta,
                                use_combined_mean=use_combined_mean)
    P = class_prototypes(s, labels_support, num_classes)

    # unary cost a_ic = ||q_i - p_c||^2  (q, P are L2-normalized -> = 2 - 2 cos)
    sim = q @ P.T                                     # [Nq, C]
    a = 2.0 - 2.0 * sim
    # softmax over -a as initial Y
    z = -a
    z = z - z.max(axis=1, keepdims=True)
    Y = np.exp(z); Y /= Y.sum(axis=1, keepdims=True)  # [Nq, C]

    # auto-select dense vs sparse knn affinity
    Nq = q.shape[0]
    if Nq > 500:
        W = _knn_affinity_sparse(q, k=knn, sigma=sigma)   # sparse [Nq, Nq]
    else:
        W = _knn_affinity(q, k=knn, sigma=sigma)           # dense  [Nq, Nq]

    for _ in range(n_iter):
        # bound-optimization update:
        # log Y_new ∝ -a + 2λ W Y
        u = -a + 2.0 * lam * (W @ Y)          # works for both dense & sparse
        u = u - u.max(axis=1, keepdims=True)
        Y = np.exp(u); Y /= (Y.sum(axis=1, keepdims=True) + 1e-12)

    preds = Y.argmax(axis=1)
    if return_probs:
        return preds, Y
    return preds


# --------- PT-MAP (Hu et al., ECCV 2022) ---------

def _log_sinkhorn(log_a: np.ndarray, n_iter: int = 10,
                  target_col: float | None = None) -> np.ndarray:
    """Sinkhorn normalization in log-space.

    Enforces:
      - row-sums = 1  (each query has a probability distribution over classes)
      - col-sums = target_col  (balanced class assignment)

    When target_col is None the column constraint is skipped (plain row-softmax).
    """
    Nq, C = log_a.shape
    if target_col is None:
        target_col = Nq / C
    log_tc = math.log(target_col)

    for _ in range(n_iter):
        # row normalise
        log_a = log_a - log_a.max(1, keepdims=True) - np.log(
            np.exp(log_a - log_a.max(1, keepdims=True)).sum(1, keepdims=True) + 1e-30
        )
        # column normalise toward target_col
        log_col = log_a.max(0, keepdims=True) + np.log(
            np.exp(log_a - log_a.max(0, keepdims=True)).sum(0, keepdims=True) + 1e-30
        )
        log_a = log_a - log_col + log_tc
    return log_a


def pt_map(feat_support: np.ndarray, labels_support: np.ndarray,
           feat_query: np.ndarray, num_classes: int = 5,
           tukey_beta: float = 0.5,
           n_iter: int = 20,
           lambda_s: float = 10.0,
           use_sinkhorn: bool = True,
           sinkhorn_iter: int = 10,
           return_probs: bool = False):
    """PT-MAP: Power Transform + MAP inference with soft-EM.

    Key differences from Mahalanobis / SimpleShot:
      1. Soft-EM jointly re-estimates class means using BOTH support (hard) and
         query (soft) samples — means drift toward the actual test distribution.
      2. Optional Sinkhorn enforces balanced class assignment, which is well-
         suited to balanced test sets (10/class in frozen_test).
      3. No covariance matrix needed — works purely with cosine distances.

    Reference: Hu et al. "Leveraging the Feature Distribution in Transfer-based
    Few-Shot Learning", ECCV 2022.  Also used in iLPC / BD-CSPN pipelines.

    Args:
      lambda_s:      temperature (= 1/(2σ²) in Gaussian model); higher = sharper.
      use_sinkhorn:  enforce balanced predictions (True for balanced test).
      sinkhorn_iter: Sinkhorn iterations per EM step.
    """
    s, q = center_and_normalize(feat_support, feat_query,
                                tukey_beta=tukey_beta,
                                use_combined_mean=True)
    D = s.shape[1]

    # ── Initialise class means from labeled support ──
    mu = np.zeros((num_classes, D), dtype=np.float64)
    for c in range(num_classes):
        m = labels_support == c
        if m.any():
            mu[c] = s[m].mean(0)
    # L2-normalise once before loop
    mu /= (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-12)

    s64 = s.astype(np.float64)
    q64 = q.astype(np.float64)

    for _ in range(n_iter):
        # ── E-step: Gaussian log-likelihood ──
        # ||q - mu||² = 2 − 2·(q·mu)  because q, mu are L2-normalised
        sim = q64 @ mu.T                          # [Nq, C]
        log_p = -lambda_s * (2.0 - 2.0 * sim)    # [Nq, C]

        if use_sinkhorn:
            log_p = _log_sinkhorn(log_p, n_iter=sinkhorn_iter,
                                  target_col=q64.shape[0] / num_classes)
        else:
            log_p = log_p - log_p.max(1, keepdims=True)
            log_p -= np.log(np.exp(log_p).sum(1, keepdims=True) + 1e-30)

        soft = np.exp(log_p)                       # [Nq, C]

        # ── M-step: update means using support (hard) + query (soft) ──
        for c in range(num_classes):
            m = labels_support == c
            w_s = float(m.sum())
            w_q = soft[:, c].sum()
            total = w_s + w_q
            if total < 1e-9:
                continue
            contrib_s = s64[m].sum(0) if m.any() else np.zeros(D)
            contrib_q = (soft[:, c:c + 1] * q64).sum(0)
            mu[c] = (contrib_s + contrib_q) / total

        # Re-normalise means after M-step
        mu /= (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-12)

    # ── Final assignment ──
    sim_f = q64 @ mu.T
    log_f = -lambda_s * (2.0 - 2.0 * sim_f)
    if use_sinkhorn:
        log_f = _log_sinkhorn(log_f, n_iter=sinkhorn_iter,
                              target_col=q64.shape[0] / num_classes)
    preds = log_f.argmax(axis=1)

    if return_probs:
        soft_f = np.exp(log_f - log_f.max(1, keepdims=True))
        soft_f /= soft_f.sum(1, keepdims=True)
        return preds, soft_f.astype(np.float32)
    return preds


# --------- alpha-TIM (Veilleux et al., NeurIPS 2021) ---------

def alpha_tim(feat_support: np.ndarray, labels_support: np.ndarray,
              feat_query: np.ndarray, num_classes: int = 5,
              tukey_beta: float = 0.5,
              use_combined_mean: bool = True,
              n_iter: int = 1000, lr: float = 1e-4,
              temperature: float = 15.0,
              alpha: float = 2.0,
              lambda_marg: float = 1.0,
              lambda_cond: float = 0.1) -> np.ndarray:
    """alpha-TIM: TIM with alpha-Renyi divergence instead of KL.

    Standard TIM maximises H(p̄_query) − E[H(p_i)], which implicitly assumes
    uniform class prior. For a class-IMBALANCED test set, this hurts minority
    classes.  alpha-TIM replaces KL with the alpha-divergence:

        D_α(p‖u) = 1/(α−1) · [Σ_c p_c^α / u_c^(α−1) − 1]

    With α > 1 (default 2.0) the divergence penalises concentration MORE
    aggressively, which pushes predictions toward balance — beneficial for
    the teacher's large imbalanced test set where a plain head under-predicts
    minority classes.

    For α → 1 the objective reduces to standard TIM.

    Reference: Veilleux et al. "Realistic Evaluation of Transductive Few-Shot
    Methods", NeurIPS 2021.
    """
    s, q = center_and_normalize(feat_support, feat_query,
                                tukey_beta=tukey_beta,
                                use_combined_mean=use_combined_mean)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = torch.tensor(s, dtype=torch.float32, device=device)
    Q = torch.tensor(q, dtype=torch.float32, device=device)
    y_S = torch.tensor(labels_support, dtype=torch.long, device=device)

    # Initialise prototypes from labeled support means
    W = torch.zeros(num_classes, S.shape[1], device=device)
    for c in range(num_classes):
        m = y_S == c
        if m.any():
            W[c] = S[m].mean(0)
    W = F.normalize(W, dim=1).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([W], lr=lr)

    for _ in range(n_iter):
        opt.zero_grad()
        Wn = F.normalize(W, dim=1)
        logits_s = temperature * (S @ Wn.T)
        logits_q = temperature * (Q @ Wn.T)

        # Support CE loss
        loss_ce = F.cross_entropy(logits_s, y_S)

        prob_q = F.softmax(logits_q, dim=1)           # [Nq, C]
        p_bar = prob_q.mean(0).clamp(1e-12, 1.0)     # marginal distribution

        if abs(alpha - 1.0) < 1e-4:
            # α→1: recover standard TIM entropy
            h_div = -(p_bar * p_bar.log()).sum()
        else:
            # alpha-Renyi divergence D_α(p̄ ‖ uniform)
            # D_α = 1/(α-1) * log( C^(α-1) * Σ p_c^α )
            h_div = (1.0 / (alpha - 1.0)) * torch.log(
                (num_classes ** (alpha - 1.0)) * (p_bar ** alpha).sum()
            )

        # Conditional entropy (per-sample)
        h_cond = -(prob_q.clamp(1e-12, 1.0) * prob_q.clamp(1e-12, 1.0).log()).sum(1).mean()

        # Maximise diversity (h_div) and minimise per-sample uncertainty (h_cond)
        loss = loss_ce - lambda_marg * h_div + lambda_cond * h_cond
        loss.backward()
        opt.step()

    with torch.no_grad():
        Wn = F.normalize(W, dim=1)
        preds = (Q @ Wn.T).argmax(1).cpu().numpy()
    return preds
