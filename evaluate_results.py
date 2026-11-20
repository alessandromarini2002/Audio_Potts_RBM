import os
import torch
import random
import hashlib
import numpy as np
from pathlib import Path
import librosa
import soundfile as sf
from scipy.spatial.distance import cdist, pdist
from frechet_audio_distance import FrechetAudioDistance
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
from transformers import AutoFeatureExtractor, ASTForAudioClassification

# Definisci i percorsi
REAL_AUDIO_DIR = "DATA/archive/happy/"
GENERATED_BASE_DIR = "RUNS/RBM_WavTokenizer/v1_003/generated_wavs/"
EPOCHS_TO_TEST = [200,400,600]

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

""" Calcolo hash del file audio """
def compute_file_md5(file_path: Path) -> str:
  # Inizializza l'oggetto per l'hash MD5
  hasher = hashlib.md5() 
  with file_path.open("rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
      hasher.update(chunk)
  return hasher.hexdigest()

# Calcola una sola volta l'insieme degli hash MD5 di tutti i file reali normalizzati
real_md5_set = {compute_file_md5(f) for f in Path(REAL_AUDIO_DIR).glob("*.wav")}

"""Caricamento Modello AST: Audio Spectrogram Transformer"""
AST_MODEL_NAME = "MIT/ast-finetuned-speech-commands-v2"
ast_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ast_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_NAME)
ast_model = ASTForAudioClassification.from_pretrained(AST_MODEL_NAME).to(
    ast_device
)
ast_model.eval()

def compute_ast_target_prob(audio_path, target_class="happy"):
  wav, sr = librosa.load(audio_path, sr=16000)
  # Converte l'audio in tensori PyTorch per AST
  inputs = ast_extractor(wav, sampling_rate=16000, return_tensors="pt")
  inputs = {k: v.to(ast_device) for k, v in inputs.items()}
  with torch.no_grad():
    logits = ast_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
  # Mappa nomi classi (minuscoli) -> ID numerico
  label2id = {k.lower(): v for k, v in ast_model.config.label2id.items()}
  # Recupera l'ID della classe target (es. "happy")
  target_id = label2id[target_class.lower()]
  return float(probs[target_id].item())

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

    internal_diversity = 0.0  # Diversità interna tra gli audio generati
    avg_nn_distance = 0.0  # Distanza media dal vicino più prossimo nel dataset reale
    exact_copy_pct = 0.0  # Percentuale di copie esatte basata sulla distanza sulle feature
    mean_dnsmos_ovrl = 0.0  # Media del punteggio di qualità percettiva DNSMOS OVRL
    mean_ast_prob = 0.0  # Media della probabilità AST per la classe target
    dataset_copy_count = 0  # Numero di file generati identici (MD5) a un file del dataset reale
    dataset_copy_md5_pct = 0.0  # Percentuale di file generati che sono copie MD5 del dataset
    internal_dup_count = 0  # Numero di file generati duplicati (MD5) tra loro nella stessa epoca
    internal_dup_pct = 0.0  # Percentuale di duplicati interni MD5 rispetto ai generati
    unique_gen_count = 0  # Numero di file generati distinti (MD5 univoci)

    if real_files and gen_files:
        real_features = np.array([extract_simple_features(str(f)) for f in real_files])
        gen_features = np.array([extract_simple_features(str(f)) for f in gen_files])

        real_md5_set = {compute_file_md5(f) for f in real_files}
        seen_gen_md5 = set()
        for gf in gen_files:
            # Calcola l'hash MD5 del file generato
            md5_val = compute_file_md5(gf)
            # Controlla se è una copia esatta di un file reale
            if md5_val in real_md5_set:
                dataset_copy_count += 1
            if md5_val not in seen_gen_md5:
                seen_gen_md5.add(md5_val)
                # Incrementa il conteggio dei file unici
                unique_gen_count += 1
            else:
                # Incrementa il contatore dei duplicati interni
                internal_dup_count += 1
        

        # Calcola le percentuali rispetto al numero totale di file generati nell'epoca
        dataset_copy_md5_pct = (dataset_copy_count / len(gen_files)) * 100.0
        internal_dup_pct = (internal_dup_count / len(gen_files)) * 100.0

        # Diversità Interna (Self-Distance tra i generati)
        pairwise_dist = pdist(gen_features, metric='euclidean')
        internal_diversity = np.mean(pairwise_dist)

        # Distanza Nearest-Neighbor dal Training Set (Anti-memorizzazione)
        distances_matrix = cdist(gen_features, real_features, metric='euclidean')
        min_nn_distances = np.min(distances_matrix, axis=1)
        # Media delle distanze minime dal dataset reale
        avg_nn_distance = np.mean(min_nn_distances)
        # Percentuale di quasi-copie sulle feature (distanza < 1e-6)
        exact_copy_pct = float(np.mean(min_nn_distances < 1e-6) * 100.0)

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
        
        # Media complessiva DNSMOS OVRL dell'epoca
        mean_dnsmos_ovrl = np.mean(ovrl_scores) if ovrl_scores else 0.0

        ast_probs = []
        for f in gen_files:
            try:
                ast_probs.append(compute_ast_target_prob(str(f), "happy"))
            except Exception as e:
                print(f"   Errore AST sul file {f.name}: {e}")
        # Media delle probabilità AST sull'epoca
        mean_ast_prob = float(np.mean(ast_probs)) if ast_probs else 0.0
        

    print(f"=== Risultati Epoca {epoch_num} ===")
    print(f"  * FAD Score (minore = migliore qualità): {fad_score:.4f}")
    print(f"  * Diversità Interna (maggiore = evita collasso): {internal_diversity:.4f}")
    print(f"  * Distanza Nearest-Neighbor Reale (> 0 = generalizzazione): {avg_nn_distance:.4f}")
    print(f"   % Copie Esatte (d=0, 0% = nessuna memorizzazione): {exact_copy_pct:.2f}%")
    print(f"  * DNSMOS OVRL (1 a 5, maggiore = voce naturale): {mean_dnsmos_ovrl:.4f}")
    print("   AST Prob classe 'happy' (0-1, maggiore = più riconoscibile):"f" {mean_ast_prob:.4f}")
    print(f"  - Copie esatte Dataset (MD5): {dataset_copy_count}/{len(gen_files)}"f" ({dataset_copy_md5_pct:.2f}%)")
    print(f"  - Duplicati interni generati (MD5): {internal_dup_count}/{len(gen_files)}"f" ({internal_dup_pct:.2f}%) | Unici: {unique_gen_count}")
    print("-" * 60)

if __name__ == "__main__":
    print("Avvio della valutazione completa (FAD + Diversità + NN + DNSMOS)...")
    for epoch in EPOCHS_TO_TEST:
        evaluate_epoch_full(epoch)