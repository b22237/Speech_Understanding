import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import torch
import torchcodec
import pyarrow
from datasets import load_dataset

def load_audio(file_path):
    sample_rate, signal = wav.read(file_path)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    if signal.dtype != np.float32 and signal.dtype != np.float64:
        max_int_value = np.iinfo(signal.dtype).max
        signal = signal.astype(np.float32) / max_int_value
    return sample_rate, signal

def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

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
    if window_type == 'hamming':
        window = 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))
    elif window_type == 'hanning':
        window = 0.5 - 0.5 * np.cos((2 * np.pi * n) / (frame_length - 1))
    elif window_type == 'rectangular':
        window = np.ones(frame_length)
    else:
        raise ValueError("Choose 'hamming', 'hanning', or 'rectangular'.")
    return frames * window

def compute_fft_power(frames, N_FFT=512):
    mag_frames = np.absolute(np.fft.rfft(frames, n=N_FFT))
    power_frames = (1.0 / N_FFT) * (mag_frames ** 2)
    return power_frames

def get_mel_filterbank(sample_rate, N_FFT=512, n_mels=40):
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    
    bin_points = np.floor((N_FFT + 1) * hz_points / sample_rate).astype(int)
    
    fbank = np.zeros((n_mels, int(np.floor(N_FFT / 2 + 1))))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])
            
    return fbank

def apply_log_mel(power_frames, fbank):
    filter_banks = np.dot(power_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = np.log(filter_banks)
    return filter_banks

def compute_dct(filter_banks, num_ceps=13):
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_ceps]
    return mfcc

def extract_mfcc(signal, sample_rate, num_ceps=13, n_mels=40, N_FFT=512, window_type='hamming'):
    emphasized = pre_emphasis(signal)
    frames = framing(emphasized, sample_rate)
    windowed = windowing(frames, window_type=window_type)
    power_frames = compute_fft_power(windowed, N_FFT=N_FFT)
    
    fbank = get_mel_filterbank(sample_rate, N_FFT=N_FFT, n_mels=n_mels)
    log_mel_energies = apply_log_mel(power_frames, fbank)
    
    mfcc_features = compute_dct(log_mel_energies, num_ceps=num_ceps)
    
    return mfcc_features

if __name__ == "__main__":
    dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)
    
    first_sample = next(iter(dataset))
    
    my_signal = first_sample["audio"]["array"]
    sample_rate = first_sample["audio"]["sampling_rate"]
    transcript = first_sample["text"]
    
    print(f"\nSample Rate: {sample_rate} Hz")
    print(f"Transcript: '{transcript}'")
    print("Extracting MFCCs...")
    
    my_mfccs = extract_mfcc(my_signal, sample_rate)
    
    print(f"\nExtraction complete!")
    print(f"MFCC Feature Matrix Shape: {my_mfccs.shape} (Frames x Coefficients)")