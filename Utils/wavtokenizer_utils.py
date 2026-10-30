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

def extract_audio_tokens(audio_dir, model, device, target_sr=24000):
    """
    Scansiona una cartella di file audio, estrae i token discreti per ciascuno
    e restituisce un tensore della forma (n_samples, 120) pronto per la Potts RBM.
    """
    valid_extensions = ('.wav', '.mp3', '.flac', '.ogg')
    
    filenames = [os.path.abspath(os.path.join(audio_dir, f)) for f in sorted(os.listdir(audio_dir)) if f.lower().endswith(valid_extensions)]
    
    all_indices = []
    
    for audio_path in tqdm(filenames, desc="Encoding audio files with WavTokenizer"):
        try:
            # Carica e converte l'audio
            wav_tensor = get_audio_tensor(audio_path, target_sr=target_sr, device=device)
            
            if wav_tensor.ndim == 1:
                wav_tensor = wav_tensor.unsqueeze(0)
            
            # Estrae i token discreti
            discrete_code, _ = reconstruct_with_wavtokenizer(wav_tensor, model)
            
            # Appiattisce il tensore dei token per renderlo una riga e lo sposta su CPU
            all_indices.append(discrete_code.reshape(-1).cpu())
            
        except Exception as e:
            print(f"Errore durante l'elaborazione del file {audio_path}: {e}")
            
    if not all_indices:
        raise ValueError("Nessun file audio è stato elaborato con successo.")
   # Imposta la dimensione fissa richiesta dalla RBM (40 siti)
    target_length = 40
    processed_indices = []
    masks = [] # Lista per le maschere
    
    for tensor in all_indices:
        orig_len = tensor.shape[0]
        if orig_len < target_length:
            # Aggiunge padding con zeri se il tensore è più corto di 40
            pad_size = target_length - orig_len
            tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=0)
            # La maschera vale 1 per i token reali, 0 sul padding
            mask = torch.cat([torch.ones(orig_len), torch.zeros(pad_size)])
        elif orig_len > target_length:
            # Tronca se il tensore è più lungo di 48
            tensor = tensor[:target_length]
            mask = torch.ones(target_length)
        else:
            mask = torch.ones(target_length)    
            
        processed_indices.append(tensor)
        masks.append(mask)
        
    # Restituisce sia i tensori impilati con forma (n_samples, 40) che le maschere corrispondenti
    return torch.stack(processed_indices, dim=0), torch.stack(masks, dim=0)