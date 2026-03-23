import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import matplotlib.pyplot as plt

# ==========================================
# 1. DEPENDENCIES FROM STEP 3
# ==========================================
try:
    from voiced_unvoiced import detect_voiced_unvoiced, consolidate_timestamps
except ImportError:
    print("Warning: Could not import voiced_unvoiced.py. Make sure it is in the same folder.")
    # You would paste your fallback Step 3 functions here if needed.

# ==========================================
# 2. MODEL LOADING & CTC ALIGNMENT
# ==========================================
# ==========================================
# 2. MODEL LOADING & CTC ALIGNMENT (LIGHTWEIGHT FALLBACK)
# ==========================================
def load_phoneme_model(model_name="facebook/wav2vec2-base-960h"):
    """
    Loads the standard, lightweight Wav2Vec2 model (~360MB).
    Outputs standard English letters instead of IPA phonemes.
    """
    print(f"Loading {model_name} (This should be much faster!)...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return processor, model

def get_model_boundaries(signal, sample_rate, processor, model):
    """
    Runs the audio through Wav2Vec2, aligns the frames to letters, 
    and groups them into Voiced/Unvoiced segments based on alphabet rules.
    """
    # 1. Run inference to get logits
    inputs = processor(signal, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        
    # 2. Get the predicted IDs (Greedy Decoding)
    predicted_ids = torch.argmax(logits, dim=-1)[0]
    
    # 1 frame is exactly 20ms in standard Wav2Vec2
    frame_duration = 0.02 
    
    # 3. Alphabet Classification Dictionary
    # Since we are using letters instead of IPA, we map hard unvoiced English consonants
    unvoiced_letters = {'P', 'T', 'K', 'F', 'S', 'H', 'C', 'X', 'Q'}
    silence_tokens = {'[PAD]', '|'} # '|' is the word boundary/space token
    
    # 4. Map frames to Voiced/Unvoiced States
    model_flags = []
    
    for token_id in predicted_ids:
        # Decode the specific frame
        token = processor.tokenizer.convert_ids_to_tokens(token_id.item())
        
        if token in silence_tokens or token in unvoiced_letters:
            model_flags.append(0)  # Unvoiced / Silence
        else:
            # Vowels (A, E, I, O, U) and voiced consonants (B, D, G, Z, M, N, etc.)
            model_flags.append(1)  
            
    # 5. Consolidate into contiguous blocks
    boundaries = []
    current_state = model_flags[0]
    start_time = 0.0
    
    for i in range(1, len(model_flags)):
        if model_flags[i] != current_state:
            end_time = i * frame_duration
            label = "Voiced" if current_state == 1 else "Unvoiced"
            boundaries.append({"start": start_time, "end": end_time, "label": label})
            current_state = model_flags[i]
            start_time = end_time
            
    # Add final block
    end_time = len(model_flags) * frame_duration
    label = "Voiced" if current_state == 1 else "Unvoiced"
    boundaries.append({"start": start_time, "end": end_time, "label": label})
    
    return boundaries

# ==========================================
# 3. RMSE CALCULATION
# ==========================================
def calculate_rmse(manual_segments, model_segments):
    """
    Calculates the Root Mean Square Error between the manual DSP boundaries 
    and the deep learning "Ground Truth" boundaries.
    """
    # Extract all boundary timestamps from both arrays
    # (A segment has a start and an end. We extract all unique transition points)
    manual_times = [seg['start'] for seg in manual_segments] + [manual_segments[-1]['end']]
    model_times = [seg['start'] for seg in model_segments] + [model_segments[-1]['end']]
    
    squared_errors = []
    
    # For every manual boundary, find the closest model boundary (Nearest Neighbor mapping)
    for m_time in manual_times:
        closest_model_time = min(model_times, key=lambda x: abs(x - m_time))
        error = m_time - closest_model_time
        squared_errors.append(error ** 2)
        
    # Formula: RMSE = sqrt( (1/n) * sum((manual_time - model_time)^2) )
    rmse = np.sqrt(np.mean(squared_errors))
    return rmse

# ==========================================
# 4. EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    from datasets import load_dataset
    
    print("1. Loading LibriSpeech sample...")
    dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)
    first_sample = next(iter(dataset))
    signal = first_sample["audio"]["array"]
    sample_rate = first_sample["audio"]["sampling_rate"]
    
    print("2. Running Manual DSP Pipeline (Step 3)...")
    manual_flags = detect_voiced_unvoiced(signal, sample_rate, threshold=0.15)
    manual_segments = consolidate_timestamps(manual_flags, hop_size_ms=10)
    
    print("3. Running Deep Learning Pipeline (Wav2Vec2)...")
    processor, model = load_phoneme_model()
    model_segments = get_model_boundaries(signal, sample_rate, processor, model)
    
    print("4. Calculating Boundary Errors...")
    rmse = calculate_rmse(manual_segments, model_segments)
    
    print("\n" + "="*50)
    print("FINAL RESULTS FOR DELIVERABLE #4")
    print("="*50)
    print(f"Total Audio Duration:  {len(signal)/sample_rate:.3f} seconds")
    print(f"Manual Segments Found: {len(manual_segments)}")
    print(f"Model Segments Found:  {len(model_segments)}")
    print("-" * 50)
    print(f"Boundary RMSE:         {rmse:.4f} seconds")
    print(f"                       ({rmse*1000:.1f} milliseconds)")
    print("="*50)