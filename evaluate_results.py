import os
import torch
import torchaudio
from audio_metrics_utils import compute_audio_metrics, compute_cer

# Percorsi alle cartelle dei file generati e dei file originali (riferimento)
gen_folder = "PATH/TO/GENERATED/AUDIOS"  # Sostituisci con la cartella dei tuoi 10 wav generati
ref_folder = "PATH/TO/REFERENCE/AUDIOS"  # Sostituisci con la cartella degli audio originali corrispondenti

pesq_scores = []
stoi_scores = []
cer_scores = []

# Supponendo di avere i 10 file nominati ad esempio sample_0.wav, sample_1.wav, ecc.
for i in range(10):
    gen_path = os.path.join(gen_folder, f"sample_{i}.wav")
    ref_path = os.path.join(ref_folder, f"sample_{i}.wav")
    
    # Testo di riferimento originale corrispondente (se usi il CER)
    # reference_text = "Il testo esatto pronunciato nell'audio originale"

    if os.path.exists(gen_path) and os.path.exists(ref_path):
        # 1. Carica gli audio come tensori (o passa direttamente i percorsi se supportato)
        gen_audio, sr_gen = torchaudio.load(gen_path)
        ref_audio, sr_ref = torchaudio.load(ref_path)

        # 2. Calcola PESQ e STOI
        metrics = compute_audio_metrics(ref_audio, gen_audio, orig_sr=sr_ref, target_sr=16000)
        pesq_scores.append(metrics["pesq"])
        stoi_scores.append(metrics["stoi"])

        # 3. Calcola il CER (richiede il testo di riferimento)
        # cer = compute_cer(reference_text, gen_path, device="cuda")
        # cer_scores.append(cer)

        print(f"File {i} -> PESQ: {metrics['pesq']:.3f} | STOI: {metrics['stoi']:.3f}")

# Stampa le medie finali sui 10 campioni
if pesq_scores:
    print("\n--- RISULTATI MEDI SU 10 CAMPIONI ---")
    print(f"PESQ Medio: {sum(pesq_scores) / len(pesq_scores):.3f}")
    print(f"STOI Medio: {sum(stoi_scores) / len(stoi_scores):.3f}")