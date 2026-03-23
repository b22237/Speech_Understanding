import os
import torch
import torchaudio
from torch.utils.data import Dataset
from collections import defaultdict

class LibriSpeechTripletDataset(Dataset):
    def __init__(self, root="./data", download=True):
        print("Loading LibriSpeech dataset (this may take a moment to download)...")
        
        # --- FIX: Ensure the directory exists before downloading ---
        os.makedirs(root, exist_ok=True)
        # -----------------------------------------------------------
        
        self.dataset = torchaudio.datasets.LIBRISPEECH(root=root, url="dev-clean", download=download)
        # Group utterances by speaker
        self.speaker_to_indices = defaultdict(list)
        for idx, item in enumerate(self.dataset):
            waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = item
            self.speaker_to_indices[speaker_id].append(idx)
            
        # Filter out speakers with fewer than 3 utterances (we need 3 for a triplet)
        self.valid_speakers = [spk for spk, indices in self.speaker_to_indices.items() if len(indices) >= 3]
        
        # Map speaker IDs to continuous integers (0 to N-1) for the classifier
        self.spk_to_label = {spk: i for i, spk in enumerate(self.valid_speakers)}
        
        # Target sample rate for SpeechBrain ECAPA-TDNN
        self.target_sr = 16000

    def add_noise(self, waveform, noise_level=0.0):
        """Simulates an 'environment' by adding background noise."""
        if noise_level > 0:
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise
        return waveform

    def __len__(self):
        # Let's say one epoch is seeing every valid speaker 5 times
        return len(self.valid_speakers) * 5

    def __getitem__(self, idx):
        # Pick a random valid speaker
        spk_id = self.valid_speakers[idx % len(self.valid_speakers)]
        label = self.spk_to_label[spk_id]
        
        # Randomly sample 3 different utterances for this speaker
        indices = torch.randperm(len(self.speaker_to_indices[spk_id]))[:3]
        
        wav1 = self.dataset[self.speaker_to_indices[spk_id][indices[0]]][0]
        wav2 = self.dataset[self.speaker_to_indices[spk_id][indices[1]]][0]
        wav3 = self.dataset[self.speaker_to_indices[spk_id][indices[2]]][0]

        # Resample if necessary (LibriSpeech is 16kHz, but it's good practice to enforce it)
        sr = self.dataset[self.speaker_to_indices[spk_id][indices[0]]][1]
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            wav1, wav2, wav3 = resampler(wav1), resampler(wav2), resampler(wav3)

        # Ensure all waveforms are at least 3 seconds long (pad if too short)
        target_length = self.target_sr * 3
        def pad_wav(w):
            if w.shape[1] < target_length:
                w = torch.nn.functional.pad(w, (0, target_length - w.shape[1]))
            # Take exactly 3 seconds to keep batch sizes uniform
            return w[:, :target_length]

        wav1, wav2, wav3 = pad_wav(wav1), pad_wav(wav2), pad_wav(wav3)

        # APPLY ENVIRONMENTS (The core of the paper's batching strategy)
        # wav1 and wav2 share Environment A (e.g., low noise)
        env_A_noise = 0.01
        wav1 = self.add_noise(wav1, noise_level=env_A_noise)
        wav2 = self.add_noise(wav2, noise_level=env_A_noise)
        
        # wav3 gets Environment B (e.g., high noise)
        env_B_noise = 0.05
        wav3 = self.add_noise(wav3, noise_level=env_B_noise)

        # Squeeze out the channel dimension for SpeechBrain compatibility
        return wav1.squeeze(0), wav2.squeeze(0), wav3.squeeze(0), label