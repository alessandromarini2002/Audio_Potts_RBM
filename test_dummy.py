import torch
import torchaudio
import os

# 1. Genera un audio fittizio di 3 secondi a 24000 Hz (frequenza richiesta da WavTokenizer)
sample_rate = 24000
duration = 3.0  # secondi
num_samples = int(sample_rate * duration)

# Crea un segnale casuale o un'onda sinusoidale
dummy_wav = torch.randn(1, num_samples) # Rumore bianco casuale

# Salvalo temporaneamente in una cartella di test
os.makedirs("DATA", exist_ok=True)
dummy_path = "DATA/dummy_test.wav"
torchaudio.save(dummy_path, dummy_wav, sample_rate)

print(f"File audio fittizio creato con successo in: {dummy_path}")

# 2. Ora puoi testare le tue funzioni di wavtokenizer_utils.py usando questo file!
from Utils.wavtokenizer_utils import load_wavtokenizer, extract_audio_tokens

config_path = "WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
ckpt_path = "WavTokenizer/ckpt/WavTokenizer_small_600_24k_4096.ckpt" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica il modello e prova l'estrazione
model = load_wavtokenizer(config_path, ckpt_path, device=device)
dataset_tensor = extract_audio_tokens("DATA/", model, device=device)

print(f"Test superato! Shape del tensore ottenuto: {dataset_tensor.shape}")
print("I 120 token estratti:")
print(dataset_tensor[0, :120])