import os
import sys
import torch
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm

# Aggiungiamo la cartella WavTokenizer al path di Python per poter importare i suoi moduli
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WAVTOKENIZER_PATH = os.path.join(_PROJECT_ROOT, "WavTokenizer")

if _WAVTOKENIZER_PATH not in sys.path:
    sys.path.insert(0, _WAVTOKENIZER_PATH)

# Importazione ufficiale dal modulo encoder di WavTokenizer
from encoder.utils import convert_audio

# Importiamo l'architettura dal repository WavTokenizer
from decoder.pretrained import WavTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path):
    """Carica il file di configurazione YAML."""
    return OmegaConf.load(config_path)

def get_audio_tensor(audio_path, target_sr=24000, device="cuda"):
    """
    Carica un file audio e applica la funzione ufficiale convert_audio
    per portarlo a 24kHz e mono, come richiesto da WavTokenizer.
    """
    wav, sr = torchaudio.load(audio_path)
    # convert_audio gestisce automaticamente resampling e canali (mono = 1)
    wav = convert_audio(wav, sr, target_sr, 1)

    return wav.to(device)
        
   
def load_wavtokenizer(config_path, ckpt_path, device="cuda"):
    """
    Carica il modello WavTokenizer usando il metodo di fabbrica ufficiale
    da decoder.pretrained.
    """
    # Carica il modello accoppiando direttamente config e checkpoint
    model = WavTokenizer.from_pretrained0802(config_path, ckpt_path)
    
    # Sposta il modello sul dispositivo (GPU/CPU) e mettilo in modalità valutazione
    model = model.to(device)
    model.eval()
    
    return model

def reconstruct_with_wavtokenizer(wav_tensor, model):
    """
    Esegue l'encoding (ricavando i token discreti) e il successivo decoding (ricostruzione audio).
    """
    device = wav_tensor.device
    bandwidth_id = torch.tensor([0], device=device)
    
    with torch.no_grad():
        # Ottiene sia le features continue che i token discreti (da 0 a 4095)
        features, discrete_code = model.encode_infer(wav_tensor, bandwidth_id=bandwidth_id)
        
        # Ricostruisce l'onda audio partendo dalle features
        wav_rec = model.decode(features, bandwidth_id=bandwidth_id)
        
    return discrete_code, wav_rec


def extract_audio_tokens_padded(audio_dir, model, device, target_sr=24000, clip_seconds=1.0, n_tokens=40):
    """
    Encode every audio file in audio_dir after zero-padding (silence) the waveform
    to clip_seconds, so every clip yields n_tokens real codec tokens (no masks).
    Return: X_idx, (n_samples, n_tokens) long tensor of codebook indices
    """
    valid_extensions = ('.wav', '.mp3', '.flac', '.ogg')
    filenames = [os.path.join(audio_dir, f) for f in sorted(os.listdir(audio_dir)) if f.lower().endswith(valid_extensions)]
    clip_len = int(target_sr * clip_seconds)

    all_indices = []
    for audio_path in tqdm(filenames, desc="Encoding audio files with WavTokenizer"):
        wav = get_audio_tensor(audio_path, target_sr=target_sr, device=device)[..., :clip_len]
        wav = torch.nn.functional.pad(wav, (0, clip_len - wav.shape[-1]))
        discrete_code, _ = reconstruct_with_wavtokenizer(wav, model)
        codes = discrete_code.reshape(-1).cpu()
        if codes.numel() < n_tokens:
            raise ValueError(f"{audio_path}: {codes.numel()} tokens < {n_tokens} after padding")
        all_indices.append(codes[:n_tokens])

    return torch.stack(all_indices, dim=0)