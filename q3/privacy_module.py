import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F

class VoiceObfuscator(nn.Module):
    def __init__(self, sample_rate=16000):
        super(VoiceObfuscator, self).__init__()
        self.sample_rate = sample_rate

    def forward(self, waveform, target_profile="male_older"):
        """
        Transforms the audio to match a target demographic profile to 
        preserve privacy while keeping linguistic content intact.
        """
        obfuscated_wav = waveform.clone()

        # Biometric Transformation Logic
        if target_profile == "male_older":
            # Lower pitch significantly (shifts formants lower, simulating larger vocal tract/older male)
            n_steps = -4.0 
        elif target_profile == "female_younger":
            # Raise pitch (shifts formants higher, simulating smaller vocal tract/younger female)
            n_steps = 4.0
        elif target_profile == "neutral":
            # Slight shift to anonymize without a specific demographic target
            n_steps = 1.5
        else:
            n_steps = 0.0

        if n_steps != 0.0:
            # Apply pitch shift
            obfuscated_wav = F.pitch_shift(
                obfuscated_wav, 
                self.sample_rate, 
                n_steps=n_steps
            )

        return obfuscated_wav