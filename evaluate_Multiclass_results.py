import os
import torch
import torch.nn.functional as F
import random
import hashlib
import numpy as np
from pathlib import Path
import librosa
import soundfile as sf
from scipy.spatial.distance import cdist, pdist
from frechet_audio_distance import FrechetAudioDistance
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# --- 1. CONFIGURAZIONE MULTICLASSE ---
CLASSES = ["dog", "happy", "stop", "yes"] 

BASE_REAL_DIR = "DATA/archive/"
GENERATED_BASE_DIR = "RUNS/RBM_WavTokenizer/v1_multiclass_007/generated_wavs/"
EPOCHS_TO_TEST = [400, 800,1200]

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

def compute_file_md5(file_path: Path) -> str:
  hasher = hashlib.md5()
  with file_path.open("rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
      hasher.update(chunk)
  return hasher.hexdigest()

real_md5_by_class = {}
for class_name in CLASSES:
    folder_reale = Path(BASE_REAL_DIR) / class_name
    real_md5_by_class[class_name] = {compute_file_md5(f) for f in folder_reale.glob("*.wav")}
all_real_md5 = set().union(*real_md5_by_class.values())

"""Caricamento Modello AST: Audio Spectrogram Transformer"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ID del modello AST pre-addestrato su Speech Commands v2 su Hugging Face
AST_MODEL_ID = "MIT/ast-finetuned-speech-commands-v2"

ast_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_ID)
# Carica il modello di classificazione audio, lo sposta su device e lo mette in modalità valutazione (.eval())
ast_model = AutoModelForAudioClassification.from_pretrained(AST_MODEL_ID).to(device)
ast_model.eval()
# Crea un dizionario "nome_classe_minuscolo -> indice_numerico" per trovare facilmente la classe target
ast_label2id = {k.lower().strip(): v for k, v in ast_model.config.label2id.items()}


def compute_ast_target_prob(audio_path, target_class):
  """Calcola la probabilità che il file audio appartenga alla classe target secondo AST."""
  # Normalizza il nome della classe (minuscolo e senza spazi esterni) per fare match con ast_label2id
  key = target_class.lower().strip()
  wav, _ = librosa.load(audio_path, sr=16000)
  # Prepara i tensori di input tramite l'extractor e li sposta sulla stessa device del modello
  inputs = ast_extractor(wav, sampling_rate=16000, return_tensors="pt")
  inputs = {k: v.to(device) for k, v in inputs.items()}

  with torch.no_grad():
    # Ottieni i logits grezzi dal modello e applica softmax per trasformarli in probabilità (somma = 1)
    logits = ast_model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze(0)
  # Estrae e restituisce come float Python la probabilità associata alla classe target
  return float(probs[ast_label2id[key]].item())

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
    exact_copy_pct = 0.0
    mean_dnsms_ovrl = 0.0
    mean_ast_prob = 0.0
    dataset_copy_count = 0
    dataset_copy_md5_pct = 0.0
    internal_dup_count = 0
    internal_dup_pct = 0.0
    unique_gen_count = 0

    if real_files and gen_files:
        real_features = np.array([extract_simple_features(str(f)) for f in real_files])
        gen_features = np.array([extract_simple_features(str(f)) for f in gen_files])

        real_md5_set = {compute_file_md5(f) for f in real_files}
        seen_gen_md5 = set()
        for gf in gen_files:
            md5_val = compute_file_md5(gf)
            if md5_val in real_md5_set:
                dataset_copy_count += 1
            if md5_val not in seen_gen_md5:
                seen_gen_md5.add(md5_val)
                unique_gen_count += 1
            else:
                internal_dup_count += 1
        
        dataset_copy_md5_pct = (dataset_copy_count / len(gen_files)) * 100.0
        internal_dup_pct = (internal_dup_count / len(gen_files)) * 100.0
        tot_gen = len(gen_files)

        # Diversità Interna (Self-Distance tra i generati)
        if len(gen_features) > 1:
            pairwise_dist = pdist(gen_features, metric='euclidean')
            internal_diversity = np.mean(pairwise_dist)

        # Distanza Nearest-Neighbor Reale (Anti-memorizzazione)
        distances_matrix = cdist(gen_features, real_features, metric='euclidean')
        # Media delle distanze minime: indica in media quanto i file generati distano dal file reale più simile
        min_nn_distances = np.min(distances_matrix, axis=1)
        avg_nn_distance = np.mean(min_nn_distances)
        # Percentuale di copie esatte (d ≈ 0): conta i file con distanza < 1e-6 (1=True, 0=False) e converte in %
        exact_copy_pct = float(np.mean(min_nn_distances < 1e-6) * 100.0)

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

        # Calcolo Probabilità di Classificazione (AST):
        ast_probs = []
        for f in gen_files:
            try:
                # Calcola la probabilità che il file 'f' appartenga a 'class_name' e la aggiunge alla lista
                ast_probs.append(compute_ast_target_prob(str(f), class_name))
            except Exception as e:
                print(f"   Errore AST sul file {f.name}: {e}")
        # Fa la media di tutte le probabilità raccolte; se la lista è vuota, mette 0.0 per evitare errori
        mean_ast_prob = float(np.mean(ast_probs)) if ast_probs else 0.0


    # Stampa risultati formattati per la classe corrente
    print(f"   * FAD Score (minore = migliore qualità): {fad_score:.4f}")
    print(f"   * Diversità Interna (maggiore = evita collasso): {internal_diversity:.4f}")
    print(f"   * Distanza Nearest-Neighbor Reale (> 0 = generalizzazione): {avg_nn_distance:.4f}")
    print(f"   % Copie Esatte (d=0, 0% = nessuna memorizzazione): {exact_copy_pct:.2f}%")
    print(f"   * DNSMOS OVRL (1 a 5, maggiore = voce naturale): {mean_dnsms_ovrl:.4f}")
    print(f"   AST Prob classe '{class_name}' (0-1, maggiore = più riconoscibile):"f" {mean_ast_prob:.4f}")
    print(f"  [{class_name}] Copie dataset (MD5): {dataset_copy_count}/{tot_gen}"f" ({dataset_copy_md5_pct:.2f}%)")
    print(f"  [{class_name}] Duplicati interni (MD5): {internal_dup_count}/{tot_gen}"f" ({internal_dup_pct:.2f}%) | Unici: {unique_gen_count}")
    print("-" * 60)

if __name__ == "__main__":
    print("Avvio della valutazione multiclasse completa (FAD + Diversità + NN + DNSMOS)...")
    
    # Doppio ciclo: Itera su tutte le epoche e per ciascuna epoca su tutte le classi
    for epoch_num in EPOCHS_TO_TEST:
        for class_name in CLASSES:
            evaluate_epoch_class(epoch_num, class_name)