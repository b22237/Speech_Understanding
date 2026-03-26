import os
import argparse
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve


try:
    torchaudio.list_audio_backends()
except AttributeError:
    torchaudio.list_audio_backends = lambda: ["soundfile", "sox_io"]

try:
    from torchaudio import _torchcodec
    import torchaudio.backend.sox_io_backend as sox
    torchaudio.load = sox.load
except ImportError:
    pass


from speechbrain.inference.speaker import EncoderClassifier
from models import DisentanglingAutoEncoder

def compute_eer(labels, scores):
    """Computes Equal Error Rate."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

def compute_mindcf(labels, scores, p_target=0.05, c_miss=1.0, c_fa=1.0):
    """Computes minimum Detection Cost Function (minDCF) per NIST SRE."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    return np.min(dcf)

def get_audio_path(utt_id, base_dir):
    """
    Parses LibriSpeech ID to path: '84-121123-0000' -> 'base/84/121123/84-121123-0000.flac'
    """
    parts = utt_id.split('-')
    if len(parts) < 3: return None
    spk_id, chapter_id = parts[0], parts[1]
    return os.path.join(base_dir, spk_id, chapter_id, f"{utt_id}.flac")

def evaluate(checkpoint_path, trials_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Evaluation on {device} ---")

    # 1. Load Frozen Baseline
    print("Loading pre-trained ECAPA-TDNN...")
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    ecapa.eval()

    # 2. Load Trained Auto-Encoder
    print(f"Loading checkpoint: {checkpoint_path}")
    autoencoder = DisentanglingAutoEncoder().to(device)
    autoencoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
    autoencoder.eval()

    # 3. Parse Trials
    trials = []
    unique_utts = set()
    with open(trials_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                trials.append((int(parts[0]), parts[1], parts[2]))
                unique_utts.update([parts[1], parts[2]])

    # 4. Feature Extraction (Unique utterances only for speed)
    print(f"Extracting embeddings for {len(unique_utts)} files...")
    embeddings = {}
    with torch.no_grad():
        for utt in tqdm(unique_utts):
            path = get_audio_path(utt, data_dir)
            if not os.path.exists(path):
                continue
            
            signal, fs = torchaudio.load(path)
            if fs != 16000:
                signal = torchaudio.transforms.Resample(fs, 16000)(signal)
            
            # Baseline -> AutoEncoder (e_spk)
            ecapa_emb = ecapa.encode_batch(signal.to(device)).squeeze(1)
            e_spk, _, _ = autoencoder(ecapa_emb)
            embeddings[utt] = e_spk.cpu()

    # 5. Scoring
    print(f"Scoring {len(trials)} pairs...")
    labels, scores = [], []
    for label, u1, u2 in trials:
        if u1 in embeddings and u2 in embeddings:
            sim = F.cosine_similarity(embeddings[u1], embeddings[u2]).item()
            labels.append(label)
            scores.append(sim)

    # 6. Results
    labels, scores = np.array(labels), np.array(scores)
    eer = compute_eer(labels, scores)
    min_dcf = compute_mindcf(labels, scores)

    print("\n" + "="*40)
    print(f"CHECKPOINT: {os.path.basename(checkpoint_path)}")
    print(f"EER:     {eer * 100:.3f}%")
    print(f"minDCF:  {min_dcf:.5f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trials", required=True)
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.trials, args.data_dir)