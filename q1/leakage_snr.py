import numpy as np
import matplotlib.pyplot as plt

def generate_test_signal(sample_rate=16000, duration=0.05):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    f1 = 415.5 
    signal = np.sin(2 * np.pi * f1 * t)
    return signal

def apply_window(signal, window_type='rectangular'):
    N = len(signal)
    n = np.arange(N)
    
    if window_type == 'hamming':
        window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    elif window_type == 'hanning':
        window = 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
    elif window_type == 'rectangular':
        window = np.ones(N)
    else:
        raise ValueError("Unknown window type")
        
    return signal * window

def compute_power_spectrum(signal, N_FFT=1024):
    mag_spectrum = np.absolute(np.fft.rfft(signal, n=N_FFT))
    power_spectrum = (1.0 / N_FFT) * (mag_spectrum ** 2)
    return power_spectrum

def measure_leakage_and_snr(power_spectrum, main_lobe_width=5):
    peak_idx = np.argmax(power_spectrum)
    
    start_idx = max(0, peak_idx - main_lobe_width)
    end_idx = min(len(power_spectrum), peak_idx + main_lobe_width + 1)
    
    p_signal = np.sum(power_spectrum[start_idx:end_idx])
    p_total = np.sum(power_spectrum)
    p_noise = p_total - p_signal
    
    if p_noise == 0:
        p_noise = np.finfo(float).eps
        
    leakage_ratio = p_noise / p_signal
    snr_db = 10 * np.log10(p_signal / p_noise)
    
    return leakage_ratio, snr_db

if __name__ == "__main__":
    sample_rate = 16000
    N_FFT = 1024
    windows = ['rectangular', 'hamming', 'hanning']
    colors = ['red', 'blue', 'green']
    
    test_signal = generate_test_signal(sample_rate=sample_rate, duration=0.05)
    
    plt.figure(figsize=(10, 6))
    freq_bins = np.linspace(0, sample_rate / 2, (N_FFT // 2) + 1)
    
    print("-" * 65)
    print(f"{'Window Type':<15} | {'Leakage (Noise/Signal)':<25} | {'SNR (dB)':<15}")
    print("-" * 65)

    for win_type, color in zip(windows, colors):
        windowed_signal = apply_window(test_signal, window_type=win_type)
        power_spec = compute_power_spectrum(windowed_signal, N_FFT=N_FFT)
        
        leakage, snr = measure_leakage_and_snr(power_spec, main_lobe_width=4)
        
        print(f"{win_type.capitalize():<15} | {leakage:>22.4f} | {snr:>11.2f} dB")
        
        power_spec_db = 10 * np.log10(power_spec + np.finfo(float).eps)
        
        plt.plot(freq_bins, power_spec_db, label=f"{win_type.capitalize()} (SNR: {snr:.1f} dB)", 
                 color=color, alpha=0.8, linewidth=1.5)

    print("-" * 65)

    plt.title("Spectral Leakage Analysis of a 415.5 Hz Sine Wave")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectrum (dB)")
    plt.xlim(0, 1500) 
    plt.ylim(-80, np.max(power_spec_db) + 10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()