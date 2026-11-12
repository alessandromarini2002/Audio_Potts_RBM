"""
Computes empirical per-site marginal distributions of VQGAN codewords over a
dataset (one categorical distribution per grid site, over the codebook states).

Used by the RBM trainer (RBM_VQGAN_Celeba_v14.py, INIT_BIAS) to initialize the
Potts RBM visible bias `b` to the log of the empirical marginals instead of zero,
so training does not have to spend early epochs re-deriving per-site codeword
frequencies from scratch.

Can also be run standalone as a diagnostic tool: it prints summary statistics
(per-site entropy, near-unused codebook states) and saves the marginals to
disk, which is useful to later compare against the codeword usage histogram of
RBM-generated samples (mismatch there is a sign of poor mixing / mode bias).

Carlo Aironi, 2026
"""

import argparse
import os
import sys

import torch


def compute_empirical_marginals(X_idx, n_states, eps=1e-6):
    """
    X_idx: (n_samples, n_sites) tensor of codebook indices in [0, n_states).
    Returns: probs (n_sites, n_states), the empirical per-site categorical
    marginal distribution, Laplace-smoothed so no entry is exactly zero
    (needed since these are later passed through log()).

    Counts via bincount on a combined (site, state) index instead of a full
    one-hot tensor: one_hot(X_idx, n_states) would materialize a
    (n_samples, n_sites, n_states) tensor -- at real dataset scale
    (28000 samples x 256 sites x 2048 states x 4 bytes) that is ~55 GiB and
    OOMs the GPU. This never allocates more than O(n_samples * n_sites) plus
    O(n_sites * n_states), independent of n_states blowing up the big axis.
    """
    n_samples, n_sites = X_idx.shape
    site_ids = torch.arange(n_sites, device=X_idx.device).unsqueeze(0).expand(n_samples, n_sites)
    combined = site_ids * n_states + X_idx.long()  # (n_samples, n_sites), values in [0, n_sites*n_states)
    counts = torch.bincount(combined.reshape(-1), minlength=n_sites * n_states).float()
    counts = counts.reshape(n_sites, n_states) + eps
    probs = counts / counts.sum(dim=-1, keepdim=True)
    return probs


def log_bias_from_marginals(probs):
    return torch.log(probs)


def summarize_marginals(probs):
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)  # (n_sites,)
    usage = probs.mean(dim=0)  # (n_states,) average usage across sites
    return {
        "mean_entropy_per_site": entropy.mean().item(),
        "min_entropy_site": entropy.min().item(),
        "max_entropy_site": entropy.max().item(),
        "n_states_near_unused": int((usage < 1e-4).sum().item()),
    }


def build_or_load_codewords(data_dir, codewords_path, recompute, vqgan_conf, vqgan_ckpt, device):
    if recompute or not os.path.isfile(codewords_path):
        # re-encode from raw images, reusing the shared VQGAN wrapper (sibling module in Utils/)
        from vqgan_utils import load_config, load_vqgan, extract_codewords

        torch.set_grad_enabled(False)
        vq_config = load_config(vqgan_conf)
        vq_model = load_vqgan(vq_config, ckpt_path=vqgan_ckpt).to(device)
        X_idx = extract_codewords(data_dir, vq_model, device).cpu()
        torch.save(X_idx, codewords_path)
        print(f"Codewords recomputed and saved to '{codewords_path}'")
    else:
        X_idx = torch.load(codewords_path)
        print(f"Loaded precomputed codewords from '{codewords_path}'")
    return X_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute empirical VQGAN codeword marginals for RBM bias init / diagnostics.")
    parser.add_argument("--data_dir", type=str, default="/mnt/sda1/home/caironi/TDM/DATA/celeba_256")
    parser.add_argument("--codewords_path", type=str, default=None, help="Defaults to <data_dir>/X2_idx.pt")
    parser.add_argument("--recompute", action="store_true", default=False, help="Re-encode from raw images instead of using the cached codewords file")
    parser.add_argument("--vqgan_conf", type=str, default="taming-transformers/logs/faceshq_vqgan/configs/2020-11-09T13-33-36-project.yaml")
    parser.add_argument("--vqgan_ckpt", type=str, default="taming-transformers/logs/faceshq_vqgan/checkpoints/last.ckpt")
    parser.add_argument("--n_states", type=int, default=1024, help="Codebook size (must match the VQGAN codebook used to produce the codewords)")
    parser.add_argument("--out", type=str, default=None, help="Output path for marginals tensor. Defaults to <data_dir>/codeword_marginals.pt")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    codewords_path = args.codewords_path or os.path.join(args.data_dir, "X2_idx.pt")
    out_path = args.out or os.path.join(args.data_dir, "codeword_marginals.pt")

    X_idx = build_or_load_codewords(args.data_dir, codewords_path, args.recompute, args.vqgan_conf, args.vqgan_ckpt, device)

    if int(X_idx.max().item()) >= args.n_states:
        print(f"WARNING: X_idx contains indices up to {int(X_idx.max().item())}, "
              f">= --n_states={args.n_states}. Pass the correct codebook size.")

    probs = compute_empirical_marginals(X_idx, args.n_states)
    log_bias = log_bias_from_marginals(probs)

    stats = summarize_marginals(probs)
    print("Empirical codeword marginals computed:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    torch.save({"probs": probs, "log_bias": log_bias, "n_states": args.n_states}, out_path)
    print(f"Marginals saved to '{out_path}'")
