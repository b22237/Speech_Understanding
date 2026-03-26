import os
import torch
import torchaudio
from privacy_module import VoiceObfuscator

# --- Configuration ---
# Create these folders in your directory
INPUT_DIR = "examples/original"
OUTPUT_DIR = "examples/obfuscated"
SAMPLE_RATE = 16000

def setup_directories():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Directories ready. Place a few test .mp3 or .wav files into '{INPUT_DIR}/'")

def run_demo():
    setup_directories()
    
    # Initialize our PyTorch Privacy Module
    obfuscator = VoiceObfuscator(sample_rate=SAMPLE_RATE)
    
    # Find audio files in the input directory
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.wav', '.mp3'))]
    
    if not files:
        print(f"Waiting for audio! Please copy 2-3 audio files from your dataset into '{INPUT_DIR}' and run again.")
        return

    print("--- RUNNING PRIVACY PRESERVING DEMO ---")
    for file in files:
        input_path = os.path.join(INPUT_DIR, file)
        
        # 1. Load Audio
        waveform, sr = torchaudio.load(input_path)
        
        # 2. Resample if necessary (Standardize to 16kHz for ASR)
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # 3. Apply Obfuscation (Let's turn them into an 'older male' since your dataset lacks them)
        print(f"Obfuscating: {file} -> target: male_older")
        obfuscated_waveform = obfuscator(waveform, target_profile="male_older")
        
        # 4. Save the output
        output_filename = f"obfuscated_{file.split('.')[0]}.wav" # Save as wav to prevent compression artifacts
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        torchaudio.save(output_path, obfuscated_waveform, SAMPLE_RATE)
        print(f"Saved to: {output_path}")

    print("\nDemo complete! Listen to the files in the 'examples' folder to hear the biometric shift.")

if __name__ == "__main__":
    run_demo()