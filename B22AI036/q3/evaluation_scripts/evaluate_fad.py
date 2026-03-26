import os
import sys

# Try to import FAD. If it fails, provide a fallback "Proxy" calculation.
try:
    from frechet_audio_distance import FrechetAudioDistance
    USE_FAD = True
except ImportError:
    print("Warning: 'frechet_audio_distance' not installed.")
    print("Falling back to a Signal-to-Noise Ratio (SNR) proxy metric...")
    import torch
    import torchaudio
    USE_FAD = False

# --- Configuration ---
# Pointing back to the examples folder you made in Step 2
ORIGINAL_DIR = "../examples/original"
OBFUSCATED_DIR = "../examples/obfuscated"

def run_fad_evaluation():
    print("--- RUNNING FAD VALIDATION ---")
    print("Checking for Toxicity Traps & Acceptability Degradation...\n")
    
    # Initialize the FAD calculator using the VGGish model
    # (This will download a small background model the first time it runs)
    frechet = FrechetAudioDistance(
        model_name="vggish",
        sample_rate=16000,
        use_pca=False, 
        use_activation=False,
        verbose=False
    )
    
    try:
        # Calculate the distance between the original and obfuscated distributions
        fad_score = frechet.score(ORIGINAL_DIR, OBFUSCATED_DIR)
        
        print(f"✅ FAD Score: {fad_score:.4f}")
        print("\n[Interpretation]")
        if fad_score < 2.0:
            print("Excellent! The privacy transformation maintained high Acceptability.")
        elif fad_score < 5.0:
            print("Good. Minor artifacts introduced, but Acceptability is maintained.")
        else:
            print("Warning: High FAD score. You may have introduced Toxicity Traps (heavy distortion).")
            
    except Exception as e:
        print(f"Error calculating FAD: {e}")
        print("Ensure you have .wav files in both directories.")

def run_proxy_evaluation():
    """Fallback proxy metric if FAD fails to install."""
    print("--- RUNNING SNR PROXY VALIDATION ---")
    orig_files = sorted([f for f in os.listdir(ORIGINAL_DIR) if f.endswith('.wav')])
    obf_files = sorted([f for f in os.listdir(OBFUSCATED_DIR) if f.endswith('.wav')])
    
    if not orig_files or not obf_files:
        print("Error: Missing audio files in examples/ directories.")
        return
        
    for orig, obf in zip(orig_files, obf_files):
        orig_wav, _ = torchaudio.load(os.path.join(ORIGINAL_DIR, orig))
        obf_wav, _ = torchaudio.load(os.path.join(OBFUSCATED_DIR, obf))
        
        # Simple Proxy: Mean Squared Error between the waveforms
        # Lower MSE means the obfuscator didn't completely destroy the signal structure
        mse = torch.nn.functional.mse_loss(orig_wav, obf_wav).item()
        print(f"File: {orig} | Proxy Degradation Score (MSE): {mse:.6f}")
        
    print("\n[Interpretation]: Lower scores indicate fewer 'Toxicity Traps'.")

if __name__ == "__main__":
    # Check if directories exist
    if not os.path.exists(ORIGINAL_DIR) or not os.path.exists(OBFUSCATED_DIR):
        print(f"Error: Directories not found. Run this script from inside the 'evaluation_scripts' folder.")
        sys.exit(1)
        
    if USE_FAD:
        run_fad_evaluation()
    else:
        run_proxy_evaluation()