import torch
import torchaudio.transforms as T
from pesq import pesq
from pystoi import stoi
import jiwer
import whisper

# Carichiamo il modello Whisper una sola volta (es. "base" o "small")
_whisper_model = None

def get_whisper_model(name="base", device="cuda"):
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(name, device=device)
    return _whisper_model

def compute_audio_metrics(ref_audio, gen_audio, orig_sr=24000, target_sr=16000):
    """
    Calcola PESQ e STOI gestendo il resampling da WavTokenizer (24kHz) 
    alle frequenze richieste dalle metriche (es. 16kHz per PESQ).
    """
    if not isinstance(ref_audio, torch.Tensor):
        ref_audio = torch.tensor(ref_audio, dtype=torch.float32)
    if not isinstance(gen_audio, torch.Tensor):
        gen_audio = torch.tensor(gen_audio, dtype=torch.float32)

    # Ricampionamento se necessario
    if orig_sr != target_sr:
        resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
        ref_audio_resampled = resampler(ref_audio)
        gen_audio_resampled = resampler(gen_audio)
    else:
        ref_audio_resampled = ref_audio
        gen_audio_resampled = gen_audio

    ref_np = ref_audio_resampled.numpy()
    gen_np = gen_audio_resampled.numpy()

    # Calcolo PESQ
    try:
        pesq_score = pesq(target_sr, ref_np, gen_np, 'wb')
    except Exception as e:
        pesq_score = float('nan')
        
    # Calcolo STOI
    stoi_score = stoi(ref_np, gen_np, target_sr, extended=False)
    
    return {
        "pesq": pesq_score,
        "stoi": stoi_score
    }

def compute_cer(reference_text, generated_audio_path, device="cuda"):
    """
    Calcola il Character Error Rate (CER) trascrivendo l'audio generato 
    tramite Whisper e confrontandolo con il testo di riferimento originale.
    """
    model = get_whisper_model(name="base", device=device)
    
    # Trascrizione dell'audio generato con Whisper
    result = model.transcribe(generated_audio_path)
    generated_text = result["text"]
    
    # Calcolo del CER tramite jiwer
    cer_score = jiwer.cer(reference_text, generated_text)
    
    return {
        "cer": cer_score,
        "transcription": generated_text
    }