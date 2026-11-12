
import math
import os
import random
import sys

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Utils"))
from compute_codeword_marginals import compute_empirical_marginals, log_bias_from_marginals
from wavtokenizer_utils import device, load_config, load_wavtokenizer, extract_multiclass_tokens_padded

N_SITES = 40
SAMPLING_MODES = ("full_gibbs", "cleanup")


class PottsRBM:
    """
    Fully-connected RBM with categorical (Potts, one-hot) visible units and
    binary hidden units. W: (n_sites, n_states, n_hidden).

    prune_mask (n_sites, n_hidden) marks still-active (site, hidden) connection
    blocks. All-ones at init; prune_to(m) only ever removes entries and
    contrastive_divergence() re-zeros pruned blocks after every update.
    """

    def __init__(self, n_sites, n_states, n_hidden, device, seed=0, w_init="fixed", w_init_scale=0.01, num_classes=4):
        assert w_init in ("fixed", "xavier"), f"w_init must be 'fixed' or 'xavier', got {w_init!r}"
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.n_sites = n_sites
        self.n_states = n_states
        self.n_hidden = n_hidden
        self.device = device
        self.num_classes = num_classes

        w_std = w_init_scale if w_init == "fixed" else math.sqrt(2.0 / (n_sites * n_states + n_hidden))
        self.w_init, self.w_init_scale, self.w_std = w_init, w_init_scale, w_std

        # W: (sites, states, hidden) -- generalizes the W (visible, hidden) of the binary RBM
        self.W = (w_std * torch.randn(n_sites, n_states, n_hidden, generator=g)).to(device)
        # b: (sites, states) -- per-site categorical visible bias
        self.b = torch.zeros(n_sites, n_states, device=device)
        # c: (hidden,) -- as in the binary RBM
        self.c = torch.zeros(n_hidden, device=device)
        # Aggiungiamo il bias condizionale per le classi: (num_classes, sites, states)
        self.class_bias = torch.zeros(num_classes, n_sites, n_states, device=device)

        # prune_mask: (n_sites, n_hidden), 1 where the (site, hidden) connection block is
        # still active. All-ones at init (fully connected); see prune_to().
        self.prune_mask = torch.ones(n_sites, n_hidden, device=device)

        self.persistent_v = None   # PCD state

    def prune_to(self, m):
        """
        Keep only the m highest-importance (site, hidden) connection blocks per
        hidden unit (L2 norm over the n_states weights), permanently zeroing the
        rest of W. Already-pruned blocks are exactly zero, so pruning is
        monotonic -- a cut connection stays cut.
        """
        importance = self.W.norm(dim=1)                       # (n_sites, n_hidden)
        topk_idx = importance.topk(m, dim=0).indices          # (m, n_hidden)
        self.prune_mask = torch.zeros_like(self.prune_mask)
        self.prune_mask.scatter_(0, topk_idx, 1.0)
        self.W = self.W * self.prune_mask.unsqueeze(1)

    def prob_h_given_v(self, v_onehot):
        activation = torch.einsum("bsk,skh->bh", v_onehot, self.W) + self.c
        return torch.sigmoid(activation)

    def sample_h_given_v(self, v_onehot):
        p_h = self.prob_h_given_v(v_onehot)
        h = (torch.rand_like(p_h) < p_h).float()
        return p_h, h
    
    def prob_v_given_h(self, h, y=None):
        # 1. Calcolo standard con i nodi nascosti 'h' e i pesi 'self.W'
        logits = torch.einsum("bh,skh->bsk", h, self.W) + self.b.unsqueeze(0)
        
        # 2. Se viene passata un'etichetta y, sommiamo il bias condizionale
        if y is not None and hasattr(self, 'class_bias'):
            # Funziona sia che y sia un batch (tensore 1D(training)) sia che sia un singolo valore/scalare(generazione)
            logits = logits + self.class_bias[y]
            
        return torch.softmax(logits, dim=-1)

    def sample_v_given_h(self, h, y=None):
        p_v = self.prob_v_given_h(h, y=y)
        batch, n_sites, n_states = p_v.shape
        idx_flat = torch.multinomial(p_v.reshape(-1, n_states), num_samples=1).squeeze(-1)
        idx = idx_flat.reshape(batch, n_sites)
        v_onehot = F.one_hot(idx, num_classes=n_states).float()
        return p_v, idx, v_onehot
    
    def contrastive_divergence(self, v_data_onehot, lr, k, persistent, weight_decay=0.0, y=None):
        batch_size = v_data_onehot.shape[0]
        p_h0, h0 = self.sample_h_given_v(v_data_onehot)
        
        if persistent:
            if self.persistent_v is None or self.persistent_v.shape[0] != batch_size:
                self.persistent_v = v_data_onehot.clone()
            v_k_onehot = self.persistent_v
            p_vk = None
            h_k = None
            for _ in range(k):
                p_hk, h_k = self.sample_h_given_v(v_k_onehot)
                # Passiamo y anche qui, così il campionamento negativo segue la classe corretta!
                p_vk, _, v_k_onehot = self.sample_v_given_h(h_k, y=y)
            self.persistent_v = v_k_onehot.detach()
        else:
            v_k_onehot = v_data_onehot
            h_k = h0
            p_vk = None
            for _ in range(k):
                p_vk, _, v_k_onehot = self.sample_v_given_h(h_k, y=y)
                p_hk, h_k = self.sample_h_given_v(v_k_onehot)

        # Gradienti standard
        grad_W = (torch.einsum("bsk,bh->skh", v_data_onehot, p_h0) - torch.einsum("bsk,bh->skh", v_k_onehot, p_hk)) / batch_size
        grad_b = (v_data_onehot - v_k_onehot).mean(dim=0)
        grad_c = (p_h0 - p_hk).mean(dim=0)

        recon_error = ((v_data_onehot - p_vk) ** 2).mean()

        # --- AGGIORNAMENTO DEI PESI STANDARD ---
        self.W += lr * (grad_W - weight_decay * self.W)
        self.W = self.W * self.prune_mask.unsqueeze(1)
        self.b += lr * grad_b
        self.c += lr * grad_c

        # --- AGGIORNAMENTO DEL CLASS_BIAS (CONDIZIONALE) ---
        if y is not None and hasattr(self, 'class_bias'):
            for c_idx in range(self.num_classes):
                mask = (y == c_idx)
                if mask.any():
                    # Calcoliamo la differenza solo per i campioni di questa specifica classe nel batch
                    grad_class_b = (v_data_onehot[mask] - v_k_onehot[mask]).mean(dim=0)
                    self.class_bias.data[c_idx] += lr * grad_class_b

        return recon_error.item()
    
    def free_energy(self, v_onehot, y=None):
        # 1. Contributo del bias visivo standard
        term_visible = torch.einsum("bsk,sk->b", v_onehot, self.b)
        
        # 2. Contributo del bias condizionale di classe (se presente e sicuro)
        term_class = 0.0
        if y is not None and hasattr(self, 'class_bias'):
            if isinstance(y, torch.Tensor) and y.ndim > 0:
                if y.numel() > 0:
                    term_class = torch.sum(v_onehot * self.class_bias[y], dim=(1, 2))
            else:
                term_class = torch.sum(v_onehot * self.class_bias[y], dim=(1, 2))

        # 3. Contributo dei nodi nascosti
        wx_c = torch.einsum("bsk,skh->bsh", v_onehot, self.W) + self.c.unsqueeze(0)
        term_hidden = F.softplus(wx_c).sum(dim=(1, 2))
        
        # L'energia totale (aggiungendo il termine di classe)
        fe = -term_visible - term_class - term_hidden
        return fe.mean()
        
def build_used_vocab(X_idx, n_states_full):
    """
    Reduced vocabulary made of the codewords that occur at least once in X_idx.
    Return: kept (n_used,) original ids in increasing order, compact_id (n_states_full,)
    original id -> compact id in [0, n_used) (-1 for unused codewords).
    """
    counts = torch.bincount(X_idx.reshape(-1), minlength=n_states_full)
    kept = (counts > 0).nonzero().squeeze(1)
    compact_id = torch.full((n_states_full,), -1, dtype=torch.long)
    compact_id[kept] = torch.arange(kept.numel())
    return kept, compact_id


def init_bias_from_marginals(rbm, X_idx, n_states, eps=1e-6):
    """Set the visible bias b to log of the empirical per-site codeword marginals."""
    probs = compute_empirical_marginals(X_idx, n_states, eps=eps)
    rbm.b = log_bias_from_marginals(probs).to(rbm.device)


def cubic_prune_schedule(epoch, warmup_epochs, prune_end_epoch, prune_every, n_start, m_final):
    """
    Gradual magnitude pruning (Zhu & Gupta, 2017-style cubic decay). Returns the
    target M for this epoch if a pruning event falls here, else None:
      - epoch < warmup_epochs or epoch > prune_end_epoch: None
      - otherwise, every prune_every epochs: m(t) = m_final + (n_start - m_final) * (1 - t)**3
        with t = (event index) / (total events).
    """
    if epoch < warmup_epochs or epoch > prune_end_epoch:
        return None
    if (epoch - warmup_epochs) % prune_every != 0:
        return None
    events = list(range(warmup_epochs, prune_end_epoch + 1, prune_every))
    t = (events.index(epoch) + 1) / len(events)
    return round(m_final + (n_start - m_final) * (1 - t) ** 3)


def train(rbm, X_idx, n_epochs, batch_size, lr, k, persistent, n_states,
          weight_decay=0.0, writer=None, prune_schedule=None, y_train=None, X_val=None, y_val=None, on_epoch_end=None):

    #Fixed lr and batch_size throughout (no schedule for either). prune_schedule:
    #dict for cubic_prune_schedule() (None disables pruning).£
    #Include X_val per il monitoraggio della validazione e prevenzione overfitting.
    #on_epoch_end(epoch): optional callback (periodic checkpoint + audio generation).
    n = X_idx.shape[0]
    current_m = prune_schedule["n_start"] if prune_schedule is not None else rbm.n_sites

    ### fixed site-wise shuffle of the val set: same per-site marginals, inter-site structure destroyed
    if X_val is not None:
        g = torch.Generator(device="cpu").manual_seed(0)
        X_val_shuf = torch.stack([X_val[torch.randperm(X_val.shape[0], generator=g).to(X_val.device), s]
                                  for s in range(X_val.shape[1])], dim=1)

    for epoch in range(n_epochs):
        ### pruning event
        if prune_schedule is not None:
            target_m = cubic_prune_schedule(epoch, **prune_schedule)
            if target_m is not None and target_m != current_m:
                rbm.prune_to(target_m)
                current_m = target_m
                print(f"[epoch {epoch + 1}] pruned to M={target_m} connections/hidden unit")
                if writer is not None:
                    writer.add_scalar("train/active_connections", target_m, epoch)

        ### one epoch of (P)CD over shuffled minibatches (Training)
        # Shuffle sincronizzato di X_idx e y_train
        perm = torch.randperm(X_idx.shape[0], device=X_idx.device)
        X_shuffled = X_idx[perm]
        y_shuffled = y_train[perm] if y_train is not None else None

        errors = []
        for start in tqdm(range(0, n, batch_size), desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False):
            batch_indices = X_shuffled[start:start + batch_size].to(X_idx.device)
            batch_onehot = F.one_hot(batch_indices, num_classes=n_states).float()

            # Estraiamo le etichette del batch corrispondenti
            batch_y = y_shuffled[start:start + batch_size].to(X_idx.device) if y_shuffled is not None else None
            errors.append(rbm.contrastive_divergence(batch_onehot,lr=lr, k=k,
                                                     persistent=persistent, weight_decay=weight_decay,y=batch_y))
        mean_err = float(np.mean(errors))
        w_norm = rbm.W.norm().item()
        print(f"[epoch {epoch + 1}/{n_epochs}] recon_error {mean_err:.6f} | ||W|| {w_norm:.3f}", flush=True)
        if writer is not None:
            writer.add_scalar("train/recon_error", mean_err, epoch)
            writer.add_scalar("weights/W_norm", w_norm, epoch)

        ### --- VALIDATION STEP ---
        if X_val is not None:
            with torch.no_grad():
                val_errors = []
                n_val = X_val.shape[0]
                for start_v in range(0, n_val, batch_size):
                    batch_val = X_val[start_v:start_v + batch_size].to(X_val.device)
                    batch_val_onehot = F.one_hot(batch_val, num_classes=n_states).float()
                    batch_val_y = y_val[start_v:start_v + batch_size].to(X_val.device) if y_val is not None else None
                    
                    val_err = rbm.contrastive_divergence(batch_val_onehot, lr=0.0, k=k,
                                                         persistent=False, weight_decay=0.0, y=batch_val_y)
                    val_errors.append(val_err)
                
                mean_val_err = float(np.mean(val_errors))
                if writer is not None:
                    writer.add_scalar("val/recon_error", mean_val_err, epoch)

        # --- CALCOLO ENERGIA LIBERA DI VALIDAZIONE ---
        if X_val is not None:
            val_fe_list = []
            for start_in in range(0, n_val, batch_size):
                batch_val = X_val[start_in:start_in + batch_size].to(X_val.device)
                batch_val_onehot = F.one_hot(batch_val, num_classes=n_states).float()
                batch_val_y = y_val[start_in:start_in + batch_size].to(X_val.device) if y_val is not None else None
                
                fe_val = rbm.free_energy(batch_val_onehot, y=batch_val_y).item()
                val_fe_list.append(fe_val)
                
            mean_val_fe = float(np.mean(val_fe_list))
            if writer is not None:
                writer.add_scalar('Free_Energy/val', mean_val_fe, epoch)

            # Per il calcolo dell'energy gap usiamo l'intero set o passiamo correttamente le etichette
            shuf_onehot = F.one_hot(X_val_shuf, num_classes=n_states).float()
            
            # Calcoliamo il gap a livello di batch o sull'intero tensore y_val
            fe_gap_list = []
            for start_in in range(0, n_val, batch_size):
                b_val_shuf = shuf_onehot[start_in:start_in + batch_size]
                b_val_orig = F.one_hot(X_val[start_in:start_in + batch_size], num_classes=n_states).float()
                b_val_y = y_val[start_in:start_in + batch_size].to(X_val.device) if y_val is not None else None
                
                gap_batch = rbm.free_energy(b_val_shuf, y=b_val_y).item() - rbm.free_energy(b_val_orig, y=b_val_y).item()
                fe_gap_list.append(gap_batch)
                
            fe_gap = float(np.mean(fe_gap_list))
            print(f"[epoch {epoch + 1}/{n_epochs}] FE gap shuffled-real (val) {fe_gap:.3f}", flush=True)
            if writer is not None:
                writer.add_scalar('Free_Energy/gap_shuffled_minus_real_val', fe_gap, epoch)
            
        # --- CALCOLO ENERGIA LIBERA DI TRAINING (sul batch corrente) ---
        sample_batch = X_idx[:batch_size].to(X_idx.device)
        sample_y = y_train[:batch_size].to(X_idx.device) if y_train is not None else None
        train_onehot = F.one_hot(sample_batch, num_classes=n_states).float()
        train_fe = rbm.free_energy(train_onehot, y=sample_y).item()
        if writer is not None:
            writer.add_scalar('Free_Energy/train', train_fe, epoch)    

        # Monitoraggio dei pesi e delle attivazioni (ogni 10 epoche)
        if epoch % 10 == 0:
            writer.add_histogram("weights/W", rbm.W, epoch)
            if hasattr(rbm, 'c') and rbm.c is not None:
                writer.add_histogram("hidden/bias_c", rbm.c, epoch)

        if on_epoch_end is not None:
            on_epoch_end(epoch)

def save_ckpt(rbm, path, kept_ids):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "w": rbm.W.cpu(), 
        "b": rbm.b.cpu(), 
        "c": rbm.c.cpu(),
        "class_bias": rbm.class_bias.cpu() if hasattr(rbm, "class_bias") and rbm.class_bias is not None else None,
        "num_classes": getattr(rbm, "num_classes", 1),
        "prune_mask": rbm.prune_mask.cpu(), 
        "kept_ids": kept_ids.cpu(),
        "n_sites": rbm.n_sites, 
        "n_states": rbm.n_states, 
        "n_hidden": rbm.n_hidden,
    }, path)
    print(f"Checkpoint saved in '{path}'")


def load_ckpt(path, device):
    """Return (rbm, kept_ids): kept_ids are the original codebook ids of the Potts states."""
    ckpt = torch.load(path, map_location=device)
    rbm = PottsRBM(n_sites=ckpt["n_sites"], n_states=ckpt["n_states"], n_hidden=ckpt["n_hidden"], num_classes=ckpt["num_classes"], device=device)
    rbm.W = ckpt["W"].to(device)
    rbm.b = ckpt["b"].to(device)
    rbm.c = ckpt["c"].to(device)
    rbm.class_bias = ckpt["class_bias"].to(device)
    rbm.prune_mask = ckpt.get("prune_mask", torch.ones(rbm.n_sites, rbm.n_hidden)).to(device)
    print(f"Checkpoint loaded from '{path}'")
    return rbm, ckpt["kept_ids"]


def decode_audio_tokens(kept_ids, idx, model, device):
    """
    Decodifica i 40 token discreti generati dalla RBM (1 secondo)
    direttamente nell'onda audio ricostruita tramite il WavTokenizer.
    idx are Potts states: mapped back to original codebook ids via kept_ids.
    """
    # Assicura che l'indice sia un vettore piatto sul device corretto
    idx = kept_ids.to(device)[idx.reshape(-1).to(device)]
    print(f"-> Shape dei token RBM in ingresso: {idx.shape}")
    print(f"-> Valori dei token in uscita dalla RBM: {idx}")
    
    # Porta i token nella forma bidimensionale (K, L) richiesta dal metodo ufficiale, es. [1, num_token]
    sample_idx = idx.unsqueeze(0)
    
    # Utilizza il metodo nativo codes_to_features per ottenere i vettori continui
    zn = model.codes_to_features(sample_idx)
    print(f"-> Shape dei vettori continui ufficiali: {zn.shape}")
    
    bandwidth_id = torch.tensor([0], device=device)
    decoded_audio = model.decode(zn, bandwidth_id=bandwidth_id)
    
    # Controllo di ampiezza / normalizzazione finale per evitare clipping
    max_val = torch.abs(decoded_audio).max()
    if max_val > 1.0:
        decoded_audio = decoded_audio / max_val * 0.99
        
    return decoded_audio


def _sample_visible_step(rbm, h, n_states, t, n_steps, sampling, y=None):
    """One visible update in a Gibbs chain. 'cleanup' takes the argmax on the
    last step; otherwise (and always for 'full_gibbs') it samples."""
    if sampling == "cleanup" and t == n_steps - 1:
        p_v = rbm.prob_v_given_h(h, y=y)
        idx = p_v.argmax(dim=-1)
        return p_v, idx, F.one_hot(idx, num_classes=n_states).float()
    return rbm.sample_v_given_h(h,y=y)


def gibbs_chain_potts(rbm, n_sites, n_states, n_steps, seed, sampling="full_gibbs", y=None):
    """Single Gibbs chain. Returns (snapshots, energies): ~10 index snapshots and the free energy at each."""
    assert sampling in SAMPLING_MODES, f"sampling must be one of {SAMPLING_MODES}, got {sampling!r}"
    assert n_states == rbm.n_states, f"n_states={n_states} != rbm.n_states={rbm.n_states}"
    torch.manual_seed(seed)
    
    idx = torch.randint(0, n_states, (1, n_sites), device=rbm.device)
        
    v_onehot = F.one_hot(idx, num_classes=n_states).float()

    snapshot_every = max(1, n_steps // 10)
    snapshots = [idx.clone()]
    energies = [rbm.free_energy(v_onehot, y=y).item()]
    for t in tqdm(range(n_steps), leave=False):
        p_h, h = rbm.sample_h_given_v(v_onehot)
        p_v, idx, v_onehot = _sample_visible_step(rbm, h, n_states, t, n_steps, sampling, y=y)
        if (t + 1) % snapshot_every == 0:
            snapshots.append(idx.clone())
            energies.append(rbm.free_energy(v_onehot, y=y).item())
    return snapshots, energies

def generate_to_tb(rbm, writer, n_samples, audio_model, kept_ids, n_sites, n_states,
                   device, n_steps=500, tag_prefix="gibbs_chain", sampling="full_gibbs",
                   seeds=None, global_step=0, y=None):

    # Definiamo exp_dir ricavandolo dal log_dir del writer di TensorBoard
    exp_dir = os.path.dirname(writer.log_dir) if hasattr(writer, 'log_dir') else "."
    
    # Modalità GENERAZIONE standard da rumore casuale
    # seeds: fixed chain seeds (same samples across epochs); None = random draw
    if seeds is None:
        seeds = random.choices(range(1000), k=n_samples)
    for i, s in enumerate(seeds):
        # Estrai solo la classe specifica per questo singolo campione i-esimo
        single_y = y[i] if y is not None else None
        snapshots, energies = gibbs_chain_potts(rbm, n_sites, n_states, n_steps=n_steps, seed=s, sampling=sampling, y=single_y)

        with torch.no_grad():
            audio_waveform = decode_audio_tokens(kept_ids, snapshots[-1], audio_model, device)

            writer.add_audio(f"{tag_prefix}/seed_{s}", audio_waveform, global_step=global_step, sample_rate=24000)
            
            class_names = ["dog", "happy", "stop", "yes"]
            # Recupera l'etichetta e associa a questo specifico campione (se esiste)
            if y is not None:
                class_id = y[i].item() if isinstance(y, torch.Tensor) else y
                class_name = class_names[class_id]
            else:
                class_name = "unknown"
            
            # Crea la struttura gerarchica dentro la cartella della run: RUNS/.../v1_multiclass_004/generated_wavs/epoch_XXXX/class_name/
            wav_dir = os.path.join(exp_dir, "generated_wavs", f"epoch_{global_step:04d}", class_name)
            os.makedirs(wav_dir, exist_ok=True)
            # Nome del file all'interno della sottocartella della classe
            wav_path = os.path.join(wav_dir, f"seed_{s}.wav")

            if audio_waveform.dim() == 1:
                audio_waveform_save = audio_waveform.unsqueeze(0)
            else:
                audio_waveform_save = audio_waveform
            
            torchaudio.save(wav_path, audio_waveform_save.cpu(), sample_rate=24000)

            for step, fe in zip(np.linspace(0, n_steps, len(energies)), energies):
                writer.add_scalar(f"gen_free_energy/{tag_prefix}/epoch_{global_step}/seed_{s}", fe, int(step))

        writer.flush()
def init_experiment(config_path):
    """
    Load the YAML config, set DEVICE, create LOG_PATH/EXPERIMENT_ID/{checkpoints,
    tensorboard}/ and dump params.txt. Returns the parameter dict.
    """
    with open(config_path) as f:
        par = yaml.load(f, Loader=yaml.FullLoader)
    par["DEVICE"] = str(device)
    par.setdefault("VOCABULARY", "reduced")   # also recorded in params.txt
    if par["VOCABULARY"] not in ("full", "reduced"):
        raise ValueError(f"VOCABULARY must be 'full' or 'reduced', got {par['VOCABULARY']!r}")
    g = par["EVAL_GIBBS_STEPS"]
    if isinstance(g, bool) or not isinstance(g, int) or g < 1:
        raise ValueError(f"EVAL_GIBBS_STEPS must be a single positive integer, got {g!r}")

    exp_dir = os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"])
    if os.path.exists(exp_dir):
        raise Exception(f"Experiment folder '{exp_dir}' already exists")
    os.makedirs(os.path.join(exp_dir, "checkpoints"))
    os.makedirs(os.path.join(exp_dir, "tensorboard"))
    with open(os.path.join(exp_dir, "params.txt"), "w") as f:
        f.write(f"Experiment ID: {par['EXPERIMENT_ID']}\n")
        f.write(f"Script: {os.path.basename(__file__)}\n\n")
        for key, value in par.items():
            f.write(f"{key}: {value}\n")
    return par


def start_experiment(par):
    print(f"START EXPERIMENT {par['EXPERIMENT_ID']}")
    random.seed(par["SEED"])
    np.random.seed(par["SEED"])
    torch.manual_seed(par["SEED"])
    torch.set_grad_enabled(False)

    exp_dir = os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"])
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, "tensorboard"))

    # load WavTokenizer
    wavtokenizer_model = load_wavtokenizer(par["WAVTOKENIZER_CONF_FILE"], ckpt_path=par["WAVTOKENIZER_CKPT_FILE"]).to(device)
    # Estrazione dinamica del codebook e definizione dei parametri
    cb_full = torch.cat([vq.codebook for vq in wavtokenizer_model.feature_extractor.encodec.quantizer.vq.layers], dim=0).to(device)
    embed_dim = 512 # Dimensione fissa dello spazio latente
    n_states_full = cb_full.shape[0]   # Dimensione del vocabolario ricavata in automatico dal modello
    print(f"Shape completa del codebook: {cb_full.shape}")

    # load or extract codewords and labels for multi-class
    codewords_path = os.path.join(par["DATA_DIR"], "X_idx_multiclass.pt")
    labels_path = os.path.join(par["DATA_DIR"], "y_labels_multiclass.pt")
    
    if os.path.isfile(codewords_path) and os.path.isfile(labels_path):
        print("Loading precomputed multiclass codewords and labels...", end="")
        X_idx = torch.load(codewords_path)
        y_labels = torch.load(labels_path)
    else:
        print("Extracting multiclass audio tokens with WavTokenizer...", end="")
        X_idx, y_labels = extract_multiclass_tokens_padded(par["DATA_DIR"], wavtokenizer_model, device, n_tokens=N_SITES)
        torch.save(X_idx, codewords_path)
        torch.save(y_labels, labels_path)
    
    print(f" done, {X_idx.shape[0]} samples total.")
    # --- STAMPA DI DEBUG AL VOLO ---
    print("DEBUG SHAPE TOKEN:", X_idx.shape)

    y_labels = y_labels.to(device)

    # Definiamo num_classes
    num_classes = 4
    ### vocabulary: "reduced" = one Potts state per codeword actually used (lossless), "full" = all of them
    if par["VOCABULARY"] == "reduced":
        kept_ids, compact_id = build_used_vocab(X_idx, n_states_full)
    else:
        kept_ids = compact_id = torch.arange(n_states_full)    # identity map: one state per codebook entry
    n_states = kept_ids.numel()
    X_idx = compact_id[X_idx].to(device)
    cb = cb_full[kept_ids.to(device)]

    indices = torch.randperm(X_idx.shape[0])
    split_idx = int(0.8 * X_idx.shape[0])
    train_idx = X_idx[indices[:split_idx]]
    val_idx = X_idx[indices[split_idx:]]
    train_y = y_labels[indices[:split_idx]] if y_labels is not None else None
    val_y = y_labels[indices[split_idx:]] if y_labels is not None else None
    
    print(f"Dataset diviso: {train_idx.shape[0]} campioni di training, {val_idx.shape[0]} campioni di validazione.")
    print(f"Codebook: {n_states_full} entries, embedding dim {embed_dim} -> {n_states} Potts states (VOCABULARY: {par['VOCABULARY']})")
    print(f"Audio tokens sequence: N_SITES = {N_SITES} tokens Potts sites ({n_states} states/site)")

    ### dump derived params
    pruning_desc = (f"gradual {N_SITES}->{par['PRUNE_M_FINAL']}/hidden unit "
                    f"(warmup={par['PRUNE_WARMUP_EPOCHS']}, end={par['PRUNE_END_EPOCH']}, every={par['PRUNE_EVERY']})"
                    if par["PRUNE"] else "disabled")
    with open(os.path.join(exp_dir, "params.txt"), "a") as f:
        f.write("\n# derived\n")
        f.write(f"N_SITES: {N_SITES}\n")
        f.write(f"N_STATES_FULL: {n_states_full}\n")
        f.write(f"N_STATES: {n_states}\n")
        f.write(f"EMBED_DIM: {embed_dim}\n")
        f.write(f"W_PARAMS_DENSE: {N_SITES * n_states * par['N_HIDDEN']}\n")
        f.write(f"PRUNING: {pruning_desc}\n")

    ### define network
    rbm = PottsRBM(n_sites=N_SITES,
                   n_states=n_states,
                   n_hidden=par["N_HIDDEN"],
                   device=device,
                   seed=par["SEED"],
                   w_init=par["W_INIT"],
                   w_init_scale=par["W_INIT_SCALE"],
                   num_classes=num_classes,)

    if par["INIT_BIAS"]:
        print("Initializing visible bias from empirical codeword marginals...")
        init_bias_from_marginals(rbm, X_idx, n_states)

    ### pruning schedule (optional)
    prune_schedule = None
    if par["PRUNE"]:
        prune_schedule = dict(warmup_epochs=par["PRUNE_WARMUP_EPOCHS"],
                              prune_end_epoch=par["PRUNE_END_EPOCH"],
                              prune_every=par["PRUNE_EVERY"],
                              n_start=N_SITES,
                              m_final=par["PRUNE_M_FINAL"])

    eval_steps = par["EVAL_GIBBS_STEPS"]
    gen_seeds = random.Random(par["SEED"]).sample(range(20000), par["GENERATE_TB_PER_STEPS"])

    def periodic_(epoch):
        """Every GENERATE_EVERY epochs (and at the last one): checkpoint + audio samples to TB."""
        e = epoch + 1
        if e % par["GENERATE_EVERY"] != 0 and e != par["N_EPOCHS"]:
            return
        save_ckpt(rbm, os.path.join(ckpt_dir, f"model_epoch_{e:04d}.pt"), kept_ids)
        print(f"Generating {len(gen_seeds)} audio samples @ epoch {e}, {eval_steps} Gibbs steps", flush=True)
        # fork the RNG so that seeding the Gibbs chains does not alter the training randomness
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
            sample_y = torch.arange(len(gen_seeds), device=device) % num_classes
            #sample_y = torch.randint(0, num_classes, (len(gen_seeds),), device=device)
            generate_to_tb(rbm, writer, len(gen_seeds), wavtokenizer_model, kept_ids, N_SITES, n_states, device,
                           n_steps=eval_steps, tag_prefix=f"gibbs_chain/{eval_steps}steps",
                           sampling=par["SAMPLING"], seeds=gen_seeds, global_step=e, y=sample_y)

    ### training loop -- fixed LR, fixed batch size
    print("Start training...")
      
    train(rbm,
         train_idx,
         n_epochs=par["N_EPOCHS"],
         batch_size=par["BATCH_SIZE"],
         lr=par["LR"],
         k=par["K_TRAIN"],
         persistent=par["PERSISTENT_CD"],
         n_states=n_states,
         weight_decay=par["WEIGHT_DECAY"],
         writer=writer,
         prune_schedule=prune_schedule,
         y_train=train_y,
         X_val=val_idx,
         y_val=val_y,
         on_epoch_end=periodic_)
    # end-of-training checkpoint and samples are produced by periodic_ at the last epoch

    writer.close()
    print("Done")


if __name__ == "__main__":
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print(f"Usage: python {os.path.basename(__file__)} <config.yaml>")
        sys.exit(1)
    start_experiment(init_experiment(sys.argv[1]))
