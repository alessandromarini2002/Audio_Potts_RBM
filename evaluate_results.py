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

# Definisci i percorsi
REAL_AUDIO_DIR = "DATA/archive/happy/"
GENERATED_BASE_DIR = "RUNS/RBM_WavTokenizer/v1_002/generated_wavs/"
EPOCHS_TO_TEST = [200,300,350,400,450,500]

target_sr = 16000
clip_seconds = 1.0
target_len = int(target_sr * clip_seconds)

files = list(Path(REAL_AUDIO_DIR).glob("*.wav"))
print(f"Elaborazione dataset reale '{REAL_AUDIO_DIR}': trovati {len(files)} file.")

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

def extract_simple_features(audio_path):
    """Estrae MFCC mediati nel tempo per calcolare le distanze in modo efficiente."""
    y, sr = librosa.load(audio_path, sr=24000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def evaluate_epoch_full(epoch_num):
    epoch_dir = os.path.join(GENERATED_BASE_DIR, f"epoch_{epoch_num:04d}")
    if not os.path.exists(epoch_dir):
        print(f"Epoch {epoch_num} folder not found: {epoch_dir}")
        return

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
            background_dir=REAL_AUDIO_DIR,
            eval_dir=epoch_dir,
            dtype="float32"
        )
    except Exception as e:
        print(f"Errore FAD Epoca {epoch_num} (possibile cartella vuota): {e}")
        return

    # Caricamento file per le metriche spaziali e DNSMOS
    real_files = list(Path(REAL_AUDIO_DIR).glob("*.wav"))
    gen_files = list(Path(epoch_dir).glob("*.wav"))

    internal_diversity = 0.0
    avg_nn_distance = 0.0
    mean_dnsmos_ovrl = 0.0

    if real_files and gen_files:
        real_features = np.array([extract_simple_features(str(f)) for f in real_files])
        gen_features = np.array([extract_simple_features(str(f)) for f in gen_files])

        # Diversità Interna (Self-Distance tra i generati)
        pairwise_dist = pdist(gen_features, metric='euclidean')
        internal_diversity = np.mean(pairwise_dist)

        # Distanza Nearest-Neighbor dal Training Set (Anti-memorizzazione)
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
            
            # Calcolo dei punteggi DNSMOS
            score_tensor = deep_noise_suppression_mean_opinion_score(wav_tensor, fs=16000, personalized=False)
            
            if isinstance(score_tensor, torch.Tensor):
                val = score_tensor.detach().cpu().numpy()
            else:
                val = np.array(score_tensor)
                
            ovrl_scores.append(float(np.mean(val)))
        
        mean_dnsmos_ovrl = np.mean(ovrl_scores) if ovrl_scores else 0.0
        

    print(f"=== Risultati Epoca {epoch_num} ===")
    print(f"  * FAD Score (minore = migliore qualità): {fad_score:.4f}")
    print(f"  * Diversità Interna (maggiore = evita collasso): {internal_diversity:.4f}")
    print(f"  * Distanza Nearest-Neighbor Reale (> 0 = generalizzazione): {avg_nn_distance:.4f}")
    print(f"  * DNSMOS OVRL (1 a 5, maggiore = voce naturale): {mean_dnsmos_ovrl:.4f}")
    print("-" * 60)

if __name__ == "__main__":
    print("Avvio della valutazione completa (FAD + Diversità + NN + DNSMOS)...")
    for epoch in EPOCHS_TO_TEST:
        evaluate_epoch_full(epoch)