import hashlib
from pathlib import Path
import shutil
import librosa
import numpy as np
import torch
from torchmetrics.functional.audio.dnsmos import (
    deep_noise_suppression_mean_opinion_score,
)
from transformers import ASTForAudioClassification, AutoFeatureExtractor

# ==========================================
# CONFIGURAZIONE
# ==========================================
EPOCH_DIR = Path("RUNS/RBM_WavTokenizer/v1_002/generated_wavs/epoch_0450")
OUT_DIR = Path("selected_samples/monoclass_v1_002_epoch450")
TARGET_CLASS = "happy"
TOP_K = 10
CANDIDATES = 40  # Numero di migliori candidati AST su cui calcolare DNSMOS
BATCH_SIZE = 16  # Dimensione batch per AST (alzabile a 32 se hai abbastanza VRAM)
AST_WEIGHT = 0.7
DNSMOS_WEIGHT = 0.3


def compute_md5(file_path: Path) -> str:
  """Calcola l'hash MD5 di un file leggendolo a blocchi."""
  hasher = hashlib.md5()
  with file_path.open("rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
      hasher.update(chunk)
  return hasher.hexdigest()


def extract_dnsmos_ovrl(wav_16k: np.ndarray) -> float:
  """Calcola il punteggio DNSMOS OVRL su un segnale mono a 16 kHz."""
  with torch.no_grad():
    audio_tensor = torch.tensor(wav_16k, dtype=torch.float32).unsqueeze(0)
    dns = deep_noise_suppression_mean_opinion_score(
        audio_tensor, fs=16000, personalized=False
    )
    dns_arr = np.asarray(
        dns.detach().cpu().numpy() if hasattr(dns, "detach") else dns
    ).flatten()
    if dns_arr.size == 0:
      return 0.0
    # L'ultimo elemento corrisponde al punteggio complessivo OVRL
    return float(dns_arr[-1])


def main() -> None:
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  for old_file in OUT_DIR.glob("*.wav"):
    old_file.unlink()

  wav_files = sorted(EPOCH_DIR.glob("*.wav"))
  if not wav_files:
    raise FileNotFoundError(f"Nessun file .wav trovato in {EPOCH_DIR}")

  # 1. Deduplicazione interna tramite hash MD5
  seen_hashes = set()
  unique_files = []
  for wav_path in wav_files:
    digest = compute_md5(wav_path)
    if digest not in seen_hashes:
      seen_hashes.add(digest)
      unique_files.append(wav_path)

  duplicates_removed = len(wav_files) - len(unique_files)
  print(
      f"File totali: {len(wav_files)} | File unici (MD5): {len(unique_files)}"
      f" | Duplicati interni rimossi: {duplicates_removed}"
  )

  # 2. Caricamento modello AST preaddestrato
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Dispositivo in uso: {device}")

  model_name = "MIT/ast-finetuned-speech-commands-v2"
  extractor = AutoFeatureExtractor.from_pretrained(model_name)
  model = ASTForAudioClassification.from_pretrained(model_name).to(device).eval()

  label2id = {k.lower(): v for k, v in model.config.label2id.items()}
  target_key = TARGET_CLASS.lower()
  if target_key not in label2id:
    raise KeyError(
        f"Classe '{TARGET_CLASS}' non trovata. Classi disponibili:"
        f" {list(label2id.keys())}"
    )
  target_id = label2id[target_key]

  # 3. Calcolo probabilità AST a batch sui soli file unici
  print("Calcolo probabilità AST a batch sui file unici...")
  ast_results = []
  for start in range(0, len(unique_files), BATCH_SIZE):
    batch_paths = unique_files[start : start + BATCH_SIZE]
    batch_audio = [librosa.load(str(p), sr=16000)[0] for p in batch_paths]

    inputs = extractor(
        batch_audio, sampling_rate=16000, padding=True, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
      probs = torch.softmax(model(**inputs).logits, dim=-1)[:, target_id]
      for p, prob in zip(batch_paths, probs.cpu().numpy()):
        ast_results.append((float(prob), p))

  # Ordina per probabilità AST decrescente e prendi i migliori CANDIDATES
  ast_results.sort(reverse=True, key=lambda x: x[0])
  top_candidates = ast_results[: min(CANDIDATES, len(ast_results))]

  # 4. Calcolo DNSMOS OVRL solo sui migliori candidati AST
  print(
      f"Calcolo DNSMOS OVRL sui migliori {len(top_candidates)} candidati"
      " AST..."
  )
  scored = []
  for ast_p, wav_path in top_candidates:
    wav_16k, _ = librosa.load(str(wav_path), sr=16000)
    dns_ovrl = extract_dnsmos_ovrl(wav_16k)
    combined_score = AST_WEIGHT * ast_p + DNSMOS_WEIGHT * (dns_ovrl / 5.0)
    scored.append((combined_score, ast_p, dns_ovrl, wav_path))

  # Ordina per punteggio combinato decrescente
  scored.sort(reverse=True, key=lambda x: x[0])

  # 5. Copia i Top K campioni unici nella cartella di destinazione
  k_actual = min(TOP_K, len(scored))
  print(
      f"\nTop {k_actual} campioni UNICI da {EPOCH_DIR} (classe"
      f" '{TARGET_CLASS}'):"
  )
  for rank, (combined_score, ast_p, dns_ovrl, src_path) in enumerate(
      scored[:k_actual], start=1
  ):
    dst_path = OUT_DIR / f"{rank:02d}_{src_path.name}"
    shutil.copy2(src_path, dst_path)
    print(
        f"{rank:2d}. {src_path.name} -> {dst_path.name} | score={combined_score:.4f}"
        f" | AST={ast_p:.4f} | DNSMOS_OVRL={dns_ovrl:.4f}"
    )


if __name__ == "__main__":
  main()