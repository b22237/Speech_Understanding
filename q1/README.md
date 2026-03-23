# Multi-Stage Cepstral Feature Extraction & Phoneme Boundary Detection

This project implements a complete manual audio processing pipeline, moving from raw digital signal processing (DSP) to modern Deep Learning validation.

## 📁 Project Structure

- `mfcc_manual.py`: The core engine. Implements Pre-emphasis, Windowing, FFT, Mel-Filterbanks, Log-compression, and DCT from scratch.
- `leakage_snr.py`: Tool to analyze spectral leakage using Rectangular, Hamming, and Hanning windows.
- `voiced_unvoiced.py`: Automated segmentation algorithm using the Real Cepstrum (High/Low Quefrency analysis).
- `phonetic_mapping.py`: Validation script using Hugging Face's `Wav2Vec2` to compute RMSE against manual boundaries.
- `q1_report.pdf`: 4-page technical report with plots and analysis.
- `data_manifest.txt`: Details of the LibriSpeech audio sample used for testing.

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.9+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt

2. Running the Pipeline
Each script is designed to be standalone for testing:

Generate MFCC Features:
python mfcc_manual.py

Analyze Windowing Leakage:
python leakage_snr.py

Detect Voiced/Unvoiced Boundaries:
python voiced_unvoiced.py

Run Deep Learning Validation (RMSE):
python phonetic_mapping.py

🛠 Troubleshooting Core Dumps
If you encounter a Core Dumped or GILState_Release error during execution (common on certain Linux/MacOS environments when mixing SciPy and Torch):

Import Order: Ensure import torch and import scipy happen at the very top of scripts.

Environment Variable: Run the script with the following flag to bypass OpenMP conflicts:

Bash

export KMP_DUPLICATE_LIB_OK=TRUE
python <script_name>.py
📊 Key Results Summary
Windowing: The Hanning window provided the highest SNR (~43.8 dB) and lowest spectral leakage.

Accuracy: The manual Cepstral boundary detection achieved an RMSE of 55.0ms compared to the Wav2Vec2 phonetic model.

