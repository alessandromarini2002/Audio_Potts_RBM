import torch
import torchaudio
import os
from Utils.wavtokenizer_utils import load_wavtokenizer, reconstruct_with_wavtokenizer, get_audio_tensor, extract_audio_tokens

# 1. Configurazione iniziale
config_path = "WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
ckpt_path = "WavTokenizer/ckpt/WavTokenizer_small_600_24k_4096.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Caricamento del WavTokenizer...")
model = load_wavtokenizer(config_path, ckpt_path, device=device)

# 2. Prendo un file audio di esempio dalla cartella DATA
audio_path ="DATA/archive/happy/0a2b400e_nohash_0.wav"
if os.path.exists(audio_path):
    print(f"Elaborazione del file: {audio_path}")
    
    # Carica e converte l'audio nel formato richiesto (es. 24kHz)
    wav_tensor = get_audio_tensor(audio_path, target_sr=24000, device=device)
    if wav_tensor.ndim == 1:
        wav_tensor = wav_tensor.unsqueeze(0)
        
    # 3. Esegui il round-trip (encode -> decode)
    discrete_code, wav_rec = reconstruct_with_wavtokenizer(wav_tensor, model)
    
    # 4. Salva il file audio ricostruito per poterlo ascoltare
    output_path = "DATA/roundtrip_test_output.wav"
    torchaudio.save(output_path, wav_rec.cpu(), 24000)
    print(f"Test completato! Ascolta il file salvato in: {output_path}")
else:
    print(f"Attenzione: Non trovo il file in {audio_path}. Inserisci un percorso valido.")

# Estrazione in batch dei token
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

