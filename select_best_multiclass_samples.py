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
# CONFIGURAZIONE MULTICLASSE (SOTTOCARTELLE PER CLASSE)
# ==========================================
EXP_NAME = "v1_multiclass_006"
EPOCH = 800 
EPOCH_DIR = Path(
    f"RUNS/RBM_WavTokenizer/{EXP_NAME}/generated_wavs/epoch_{EPOCH:04d}"
)
OUT_DIR = Path(f"selected_samples/multiclass_{EXP_NAME}_epoch{EPOCH}")

# 10 campioni per classe
PER_CLASS_QUOTA = {
    "dog": 10,
    "happy": 10,
    "stop": 10,
    "yes": 10,
}

CANDIDATES_PER_CLASS = 40
BATCH_SIZE = 16
AST_WEIGHT = 0.7
DNSMOS_WEIGHT = 0.3


def compute_md5(file_path: Path) -> str:
  hasher = hashlib.md5()
  with file_path.open("rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
      hasher.update(chunk)
  return hasher.hexdigest()


def extract_dnsmos_ovrl(wav_16k: np.ndarray) -> float:
  with torch.no_grad():
    audio_tensor = torch.tensor(wav_16k, dtype=torch.float32).unsqueeze(0)
    dns = deep_noise_suppression_mean_opinion_score(
        audio_tensor, fs=16000, personalized=False
    )
    dns_arr = np.asarray(
        dns.detach().cpu().numpy() if hasattr(dns, "detach") else dns
    ).flatten()
    return float(dns_arr[-1]) if dns_arr.size > 0 else 0.0


def main() -> None:
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  for old_file in OUT_DIR.glob("*.wav"):
    old_file.unlink()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Esperimento: {EXP_NAME} | Epoca: {EPOCH} | Dispositivo: {device}")

  model_name = "MIT/ast-finetuned-speech-commands-v2"
  extractor = AutoFeatureExtractor.from_pretrained(model_name)
  model = ASTForAudioClassification.from_pretrained(model_name).to(device).eval()
  label2id = {k.lower(): v for k, v in model.config.label2id.items()}

  for cls in PER_CLASS_QUOTA:
    if cls.lower() not in label2id:
      raise KeyError(
          f"Classe '{cls}' non trovata nel modello AST. Disponibili:"
          f" {list(label2id.keys())}"
      )

  final_selection = []
  global_selected_hashes = set()

  for cls, quota in PER_CLASS_QUOTA.items():
    if quota <= 0:
      continue

    cls_dir = EPOCH_DIR / cls
    wav_files = sorted(cls_dir.glob("*.wav"))
    if not wav_files:
      raise FileNotFoundError(f"Nessun file .wav trovato in {cls_dir}")

    # 1. Deduplicazione MD5 all'interno della cartella di classe
    seen_hashes = set()
    unique_files = []
    file_to_hash = {}
    for wav_path in wav_files:
      digest = compute_md5(wav_path)
      file_to_hash[wav_path] = digest
      if digest not in seen_hashes:
        seen_hashes.add(digest)
        unique_files.append(wav_path)

    print(
        f"\n[{cls}] Tot: {len(wav_files)} | Unici (MD5): {len(unique_files)}"
        f" | Duplicati rimossi: {len(wav_files) - len(unique_files)}"
    )

    # 2. Carica audio dei file unici e calcola probabilità AST per `cls`
    audio_map = {p: librosa.load(str(p), sr=16000)[0] for p in unique_files}
    target_id = label2id[cls.lower()]
    ast_results = []

    for start in range(0, len(unique_files), BATCH_SIZE):
      batch_paths = unique_files[start : start + BATCH_SIZE]
      batch_audio = [audio_map[p] for p in batch_paths]

      inputs = extractor(
          batch_audio, sampling_rate=16000, padding=True, return_tensors="pt"
      )
      inputs = {k: v.to(device) for k, v in inputs.items()}

      with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[:, target_id]
        for p, prob in zip(batch_paths, probs.cpu().numpy()):
          ast_results.append((float(prob), p))

    ast_results.sort(reverse=True, key=lambda x: x[0])
    top_candidates = ast_results[: min(CANDIDATES_PER_CLASS, len(ast_results))]

    # 3. Calcola DNSMOS OVRL sui migliori candidati AST della classe
    scored_cls = []
    for ast_p, wav_path in top_candidates:
      dns_ovrl = extract_dnsmos_ovrl(audio_map[wav_path])
      combined = AST_WEIGHT * ast_p + DNSMOS_WEIGHT * (dns_ovrl / 5.0)
      scored_cls.append((combined, ast_p, dns_ovrl, wav_path))

    scored_cls.sort(reverse=True, key=lambda x: x[0])

    picked = 0
    for combined, ast_p, dns_ovrl, wav_path in scored_cls:
      digest = file_to_hash[wav_path]
      if digest in global_selected_hashes:
        continue
      global_selected_hashes.add(digest)
      final_selection.append((cls, combined, ast_p, dns_ovrl, wav_path))
      picked += 1
      if picked >= quota:
        break

  # 4. Copia i file selezionati in OUT_DIR
  print(f"\nCopia di {len(final_selection)} file in {OUT_DIR}:")
  for idx, (cls, combined, ast_p, dns_ovrl, src_path) in enumerate(
      final_selection, start=1
  ):
    dst_name = f"{idx:02d}_{cls}_{src_path.name}"
    dst_path = OUT_DIR / dst_name
    shutil.copy2(src_path, dst_path)
    print(
        f"{idx:2d}. [{cls:5s}] {src_path.name} -> {dst_name} | "
        f"score={combined:.4f} | AST={ast_p:.4f} | DNSMOS_OVRL={dns_ovrl:.4f}"
    )


if __name__ == "__main__":
  main()