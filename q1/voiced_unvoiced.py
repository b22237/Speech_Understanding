import numpy as np
import matplotlib.pyplot as plt

try:
    from mfcc_manual import framing, windowing
except ImportError:
    def framing(signal, sample_rate, frame_size_ms=25, hop_size_ms=10):
        frame_length = int(round((frame_size_ms / 1000.0) * sample_rate))
        hop_length = int(round((hop_size_ms / 1000.0) * sample_rate))
        signal_length = len(signal)
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / hop_length)) + 1
        pad_signal_length = num_frames * hop_length + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(signal, z)
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
                  np.tile(np.arange(0, num_frames * hop_length, hop_length), (frame_length, 1)).T
        return pad_signal[indices.astype(np.int32, copy=False)]

    def windowing(frames, window_type='hamming'):
        frame_length = frames.shape[1]
        n = np.arange(frame_length)
        window = 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))
        return frames * window

def compute_real_cepstrum(frame, N_FFT=1024):
    mag_spectrum = np.absolute(np.fft.rfft(frame, n=N_FFT))
    log_spectrum = np.log(mag_spectrum + np.finfo(float).eps)
    cepstrum = np.fft.irfft(log_spectrum, n=N_FFT)
    return cepstrum

def detect_voiced_unvoiced(signal, sample_rate, frame_size_ms=25, hop_size_ms=10, threshold=0.15):
    frames = framing(signal, sample_rate, frame_size_ms, hop_size_ms)
    windowed_frames = windowing(frames, window_type='hamming')
    
    min_freq, max_freq = 50.0, 400.0
    min_quefrency = 1.0 / max_freq  
    max_quefrency = 1.0 / min_freq  
    
    min_bin = int(min_quefrency * sample_rate)
    max_bin = int(max_quefrency * sample_rate)
    
    voiced_flags = []
    for frame in windowed_frames:
        cepstrum = compute_real_cepstrum(frame, N_FFT=1024)
        pitch_region = cepstrum[min_bin:max_bin]
        max_peak = np.max(pitch_region)
        
        if max_peak > threshold:
            voiced_flags.append(1)
        else:
            voiced_flags.append(0)
            
    return np.array(voiced_flags)

def consolidate_timestamps(voiced_flags, hop_size_ms=10):
    boundaries = []
    current_state = voiced_flags[0]
    start_time = 0.0
    
    for i in range(1, len(voiced_flags)):
        if voiced_flags[i] != current_state:
            end_time = i * (hop_size_ms / 1000.0)
            label = "Voiced" if current_state == 1 else "Unvoiced"
            boundaries.append({"start": start_time, "end": end_time, "label": label})
            current_state = voiced_flags[i]
            start_time = end_time
            
    end_time = len(voiced_flags) * (hop_size_ms / 1000.0)
    label = "Voiced" if current_state == 1 else "Unvoiced"
    boundaries.append({"start": start_time, "end": end_time, "label": label})
    
    return boundaries

def plot_boundaries(signal, sample_rate, boundaries):
    time_axis = np.linspace(0, len(signal) / sample_rate, num=len(signal))
    
    plt.figure(figsize=(14, 5))
    plt.plot(time_axis, signal, color='black', alpha=0.6, label='Waveform')
    
    added_voiced_label = False
    added_unvoiced_label = False
    
    for b in boundaries:
        if b['label'] == 'Voiced':
            color = 'green'
            alpha = 0.3
            label = 'Voiced' if not added_voiced_label else ""
            added_voiced_label = True
        else:
            color = 'red'
            alpha = 0.2
            label = 'Unvoiced / Silence' if not added_unvoiced_label else ""
            added_unvoiced_label = True
            
        plt.axvspan(b['start'], b['end'], color=color, alpha=alpha, label=label)

    plt.title("Speech Segmentation: Voiced vs Unvoiced Regions")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.xlim(0, time_axis[-1])
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from datasets import load_dataset
    
    print("Loading LibriSpeech sample...")
    dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)
    first_sample = next(iter(dataset))
    
    signal = first_sample["audio"]["array"]
    sample_rate = first_sample["audio"]["sampling_rate"]
    
    print("Running Boundary Detection via Cepstrum...")
    flags = detect_voiced_unvoiced(signal, sample_rate, threshold=0.15)
    
    print("Consolidating Timestamps...")
    segments = consolidate_timestamps(flags)
    
    print("\nDetected Segments (First 10):")
    print(f"{'Start (s)':<10} | {'End (s)':<10} | {'Classification':<15}")
    print("-" * 40)
    for seg in segments[:10]:
        print(f"{seg['start']:<10.3f} | {seg['end']:<10.3f} | {seg['label']:<15}")
        
    print("\nGenerating Visualization...")
    plot_boundaries(signal, sample_rate, segments)