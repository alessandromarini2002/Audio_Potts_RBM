import os
import torch
import random
import numpy as np
from pathlib import Path
import librosa
import soundfile as sf
from scipy.spatial.distance import cdist, pdist
from frechet_audio_distance import FrechetAudioDistance
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score

# --- 1. CONFIGURAZIONE MULTICLASSE ---
CLASSES = ["dog", "happy", "stop", "yes"] 

BASE_REAL_DIR = "DATA/archive/"
GENERATED_BASE_DIR = "RUNS/RBM_WavTokenizer/v1_multiclass_003/generated_wavs/"
EPOCHS_TO_TEST = [400, 800, 1200]

target_sr = 16000
clip_seconds = 1.0
target_len = int(target_sr * clip_seconds)

for class_name in CLASSES:
    folder_reale = os.path.join(BASE_REAL_DIR, class_name)
    files = list(Path(folder_reale).glob("*.wav"))
    print(f"Elaborazione classe '{class_name}': trovati {len(files)} file reali.")

    for f in files:
        try:
            y, sr = librosa.load(str(f), sr=target_sr)
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)), mode='constant')
            else:
                y = y[:target_len]
            sf.write(str(f), y, target_sr)
        except Exception as e:
            print(f"   Errore con il file {f.name}: {e}")

print("Fatto! Tutti i dataset reali ora hanno file omogenei a 1 secondo.")

def extract_simple_features(audio_path):
    """Estrae MFCC mediati nel tempo per calcolare le distanze in modo efficiente."""
    y, sr = librosa.load(audio_path, sr=24000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def evaluate_epoch_class(epoch_num, class_name):
    """Esegue la valutazione completa per una specifica epoca e classe."""
    real_audio_dir = os.path.join(BASE_REAL_DIR, class_name)
    epoch_dir = os.path.join(GENERATED_BASE_DIR, f"epoch_{epoch_num:04d}", class_name)
    
    if not os.path.exists(epoch_dir):
        print(f"   [Salta] Cartella non trovata: {epoch_dir}")
        return

    if not os.path.exists(real_audio_dir):
        print(f"   [Salta] Cartella reale non trovata: {real_audio_dir}")
        return

    print(f"\n--- Valutazione | Epoca: {epoch_num} | Classe: {class_name.upper()} ---")

    # Calcolo del FAD ufficiale
    frechet = FrechetAudioDistance(
        model_name="vggish",
        sample_rate=16000,
        use_pca=False,
        use_activation=False,
        verbose=False
    )

    try:
        fad_score = frechet.score(
            background_dir=real_audio_dir,
            eval_dir=epoch_dir,
            dtype="float32"
        )
    except Exception as e:
        print(f"   Errore FAD per {class_name} (epoca {epoch_num}): {e}")
        fad_score = -1.0

    # Caricamento file per le metriche spaziali e DNSMOS
    real_files = list(Path(real_audio_dir).glob("*.wav"))
    gen_files = list(Path(epoch_dir).glob("*.wav"))

    internal_diversity = 0.0
    avg_nn_distance = 0.0
    mean_dnsms_ovrl = 0.0

    if real_files and gen_files:
        real_features = np.array([extract_simple_features(str(f)) for f in real_files])
        gen_features = np.array([extract_simple_features(str(f)) for f in gen_files])

        # Diversità Interna (Self-Distance tra i generati)
        if len(gen_features) > 1:
            pairwise_dist = pdist(gen_features, metric='euclidean')
            internal_diversity = np.mean(pairwise_dist)

        # Distanza Nearest-Neighbor Reale (Anti-memorizzazione)
        distances_matrix = cdist(gen_features, real_features, metric='euclidean')
        min_nn_distances = np.min(distances_matrix, axis=1)
        avg_nn_distance = np.mean(min_nn_distances)

        # Calcolo DNSMOS (Overall Quality) sui file generati
        ovrl_scores = []
        # Seleziona al massimo 30 file casuali per evitare rallentamenti estremi
        files_to_test = gen_files if len(gen_files) <= 30 else random.sample(gen_files, 30)

        for f in files_to_test:
            wav, sr = librosa.load(str(f), sr=16000)
            wav_tensor = torch.tensor(wav).unsqueeze(0)
            
            try:
                score_tensor = deep_noise_suppression_mean_opinion_score(wav_tensor, fs=16000, personalized=False)
                if isinstance(score_tensor, torch.Tensor):
                    val = score_tensor.detach().cpu().numpy()
                else:
                    val = np.array(score_tensor)
                ovrl_scores.append(float(np.mean(val)))
            except Exception as e:
                continue

        mean_dnsms_ovrl = np.mean(ovrl_scores) if ovrl_scores else 0.0

    # Stampa risultati formattati per la classe corrente
    print(f"   * FAD Score (minore = migliore qualità): {fad_score:.4f}")
    print(f"   * Diversità Interna (maggiore = evita collasso): {internal_diversity:.4f}")
    print(f"   * Distanza Nearest-Neighbor Reale (> 0 = generalizzazione): {avg_nn_distance:.4f}")
    print(f"   * DNSMOS OVRL (1 a 5, maggiore = voce naturale): {mean_dnsms_ovrl:.4f}")
    print("-" * 60)

if __name__ == "__main__":
    print("Avvio della valutazione multiclasse completa (FAD + Diversità + NN + DNSMOS)...")
    
    # Doppio ciclo: Itera su tutte le epoche e per ciascuna epoca su tutte le classi
    for epoch_num in EPOCHS_TO_TEST:
        for class_name in CLASSES:
            evaluate_epoch_class(epoch_num, class_name)