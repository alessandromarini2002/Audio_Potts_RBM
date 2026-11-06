import torch
import torchaudio
import os
from Utils.wavtokenizer_utils import load_wavtokenizer, reconstruct_with_wavtokenizer, get_audio_tensor, extract_audio_tokens
from RBM_latent import PottsRBM, load_ckpt

# 1. Configurazione iniziale
config_path = "WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
wavtokenizer_ckpt_path = "WavTokenizer/ckpt/WavTokenizer_small_600_24k_4096.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Caricamento del WavTokenizer...")
model = load_wavtokenizer(config_path, wavtokenizer_ckpt_path, device=device)

# 2. Estrazione in batch dei token
target_dir = "DATA/archive/happy/"
print(f"\n[Test 2] Estrazione dei token in batch dalla cartella: {target_dir}")

if os.path.exists(target_dir):
    # Estrae i token e restituisce il tensore della forma (n_samples, 40) pronto per la RBM
    dataset_tensor = extract_audio_tokens(target_dir, model, device=device)
    
    print(f"-> Test superato! Shape del tensore ottenuto: {dataset_tensor.shape}")
    print(f"-> 40 token estratti dal primo file:")
    print(dataset_tensor[0, :40])
else:
    print(f"Attenzione: La cartella {target_dir} non esiste.")

with torch.no_grad():
    # 3. DA TOKEN A VETTORI CONTINUI (Forma corretta [1, num_tokens] per il modello)
    sample_idx = dataset_tensor[4].unsqueeze(0).to(device) # shape: [1, 40]

    """
    # Carica direttamente la RBM e i token conservati usando la funzione nativa
    rbm_ckpt_path = "RUNS/RBM_WavTokenizer/08/checkpoints/model_final4.pt"
    rbm, kept_ids = load_ckpt(rbm_ckpt_path, device=device)
    n_states = rbm.n_states

    # Mappa i token reali globali sugli indici ridotti della RBM usando kept_ids
    mapped_sample_idx = torch.tensor([
        (kept_ids == val.item()).nonzero(as_tuple=True)[0].item() 
        for val in sample_idx[0]
        ], device=device).unsqueeze(0)
    
    # Convertiamo in one-hot usando i nuovi indici mappati
    v_onehot = torch.nn.functional.one_hot(mapped_sample_idx, num_classes=n_states).float()
        
    # Passaggio attraverso la RBM (1 passo di ricostruzione o denoising)
    p_h, h = rbm.sample_h_given_v(v_onehot)
    p_v = rbm.prob_v_given_h(h)
        
    # Otteniamo i nuovi token ricostruiti dalla RBM (tramite argmax o campionamento)
    recon_idx = p_v.argmax(dim=-1) # shape: [1, 40]

    # RIMAPPARE ALLO SPAZIO GLOBALE DEL WAVTOKENIZER:
    # I recon_idx sono indici locali (0..n_states), dobbiamo convertirli nei token globali originali tramite i kept_ids
    recon_idx_global = kept_ids[recon_idx]
    # Sostituiamo con i token globali corretti per la decodifica
    sample_idx = recon_idx_global
    print("-> Token passati attraverso la RBM e rimappati con successo!")

    """
    
    # Il metodo ufficiale codes_to_features gestisce internamente shape, offset e codebook
    zn = model.codes_to_features(sample_idx)
    print(f"-> Shape dei vettori continui (zn) ottenuti: {zn.shape}")

    # 4. DECODIFICA AUDIO NATIVA
    bandwidth_id = torch.tensor([0], device=device)
    decoded_audio = model.decode(zn, bandwidth_id=bandwidth_id)

    # Normalizzazione anti-clipping
    max_val = torch.abs(decoded_audio).max()
    if max_val > 1.0:
        decoded_audio = decoded_audio / max_val * 0.99
        
    # 5. Salva il file audio ricostruito
    output_path = "DATA/roundtrip_test_output_official.wav"
    torchaudio.save(output_path, decoded_audio.squeeze(0).cpu(), 24000)
    print(f"Test completato! Ascolta il file pulito salvato in: {output_path}")

    """
    # Controlliamo quanti token sono effettivamente diversi dopo il passaggio nella RBM
    num_changed = (mapped_sample_idx != recon_idx).sum().item()
    print(f"-> Token modificati dalla RBM: {num_changed} su {mapped_sample_idx.numel()}")

    """     
      

