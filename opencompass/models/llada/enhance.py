"""Global-head enhancement (GCG) for LLaDA.

For each target layer, heads are ranked by their attention center-of-mass (COM)
distance. The top ``pct`` heads form the global set G and the bottom ``pct``
heads form the local set L. Each global head output is enhanced by amplifying
the component that the local heads cannot express:

    h_g <- h_g + alpha * (h_g - P_L(h_g))

where P_L is a per-token ridge projection onto the span of the local head
outputs. The update is applied to the SDPA output, before the output
projection, so it is compatible with flash attention.
"""
import pickle

import torch


def _iter_llada_blocks(model):
    """Return the list of LLaDA transformer blocks."""
    m = model.module if hasattr(model, 'module') else model
    if hasattr(m, 'model') and hasattr(m.model, 'transformer'):
        tr = m.model.transformer
    elif hasattr(m, 'transformer'):
        tr = m.transformer
    else:
        raise RuntimeError('Cannot find LLaDA transformer blocks')
    if hasattr(tr, 'blocks'):
        return list(tr.blocks)
    raise RuntimeError('No .blocks found')


def load_com_scores(pkl_path):
    """Load per-head COM scores as {layer_idx: ndarray(n_heads,)}."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    arr = data['all_mean']  # (n_layers, n_heads)
    return {l: arr[l] for l in range(arr.shape[0])}


def project_onto_local_span(H_G, H_L, lam=1e-4):
    """Ridge projection of global head outputs onto the local head span.

    Proj(x) = A (A^T A + lam I)^{-1} A^T x, solved independently per token.

    Args:
        H_G: (B, n_g, T, D_h) global head outputs.
        H_L: (B, k,   T, D_h) local head outputs (the basis).
        lam: ridge coefficient.

    Returns:
        (B, n_g, T, D_h) projections, in the dtype of H_G.
    """
    B, k, T, D_h = H_L.shape
    n_g = H_G.shape[1]

    H_L_t = H_L.permute(0, 2, 1, 3).reshape(B * T, k, D_h).float()
    H_G_t = H_G.permute(0, 2, 3, 1).reshape(B * T, D_h, n_g).float()

    gram = torch.bmm(H_L_t, H_L_t.transpose(-1, -2))
    gram_reg = gram + lam * torch.eye(k, device=gram.device, dtype=gram.dtype)
    coeff = torch.linalg.solve(gram_reg, torch.bmm(H_L_t, H_G_t))  # (B*T, k, n_g)
    proj_t = torch.bmm(H_L_t.transpose(-1, -2), coeff)  # (B*T, D_h, n_g)

    return proj_t.reshape(B, T, D_h, n_g).permute(0, 3, 1, 2).contiguous().to(H_G.dtype)


def select_heads(com, pct):
    """Top-``pct`` heads by COM are global, bottom-``pct`` heads are local."""
    n_heads = len(com)
    n_global = max(1, int(round(n_heads * pct)))
    n_local = min(max(1, int(round(n_heads * pct))), n_heads - n_global)
    order = sorted(range(n_heads), key=lambda h: com[h], reverse=True)
    return order[:n_global], order[-n_local:]


def apply_gcg(model, layer_ids, com_scores, pct=0.25, alpha=0.3, lam=1e-4):
    """Patch the SDPA of each layer in ``layer_ids`` with the GCG update.

    The patch is permanent for the lifetime of ``model``.
    """
    blocks = _iter_llada_blocks(model)

    for lid in layer_ids:
        com = com_scores.get(lid)
        if com is None:
            continue
        g_idx, l_idx = select_heads(com, pct)
        blk = blocks[lid]
        orig = blk._scaled_dot_product_attention

        def patched(self_blk, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                    _orig=orig, _g=g_idx, _l=l_idx):
            att = _orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
            H_G = att[:, _g, :, :]
            H_L = att[:, _l, :, :]
            proj = project_onto_local_span(H_G, H_L, lam=lam)
            out = att.clone()
            out[:, _g, :, :] = H_G + alpha * (H_G - proj)
            return out

        blk._scaled_dot_product_attention = patched.__get__(blk, blk.__class__)
