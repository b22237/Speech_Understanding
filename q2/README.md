
```markdown
# Disentangled Representation Learning for Environment-agnostic Speaker Recognition

This repository contains a "justified reduced reproduction" of the framework proposed in [arXiv:2406.14559](https://arxiv.org/abs/2406.14559). The project implements a Disentangled Representation Learning (DRL) framework using an Auto-Encoder to separate speaker identity from environmental factors.

## Project Structure
- `train.py`: Main training script with support for fixed and dynamic loss weighting.
- `eval.py`: Evaluation script calculating EER and minDCF using LibriSpeech trials.
- `models.py`: Architectures for the Auto-Encoder, Discriminators, and GRL.
- `dataset.py`: Custom triplet loader using LibriSpeech with synthetic noise injection.
- `configs/`: YAML files for Baseline (fixed) and Improved (dynamic) experiments.
- `results/`: Stores model checkpoints (.pt) and loss curve plots.

## Setup & Prerequisites
1. **Environment**: Python 3.10+ is recommended.
2. **Dependencies**: Install via `pip install -r requirements.txt`.
3. **Data**: The project uses the `dev-clean` subset of LibriSpeech. The `train.py` script will automatically download this (~330MB) upon first execution.
4. **Pre-trained Baseline**: The script automatically fetches the `speechbrain/spkrec-ecapa-voxceleb` model to serve as the fixed embedding extractor.

## Implementation Details
Due to the massive compute requirements of the original VoxCeleb2 dataset (150GB+), this reproduction uses **LibriSpeech** as a proxy. We simulate "environment mismatch" by:
- Pairing two utterances with a shared low-noise profile (Environment A).
- Pairing a third utterance from the same speaker with a higher noise profile (Environment B).
- Training the Auto-Encoder to swap these "codes" and reconstruct the original speaker identity.

## Reproduction Steps

### 1. Training
Run the baseline experiment (Paper implementation):
```bash
python train.py --config configs/proposed_fixed.yaml --seed 42
```

Run the improved experiment (Part D - Dynamic Uncertainty Weighting):
```bash
python train.py --config configs/improved_dynamic.yaml --seed 42
```

### 2. Trial Generation
Before evaluation, generate the verification trials from the downloaded data:
*(A helper script to generate `test.txt` is included in the project notes).*

### 3. Evaluation
Evaluate the generated checkpoints:
```bash
python eval.py --checkpoint results/best_model_proposed_fixed_lambdas_seed42.pt --trials test.txt --data_dir ./data/LibriSpeech/dev-clean
```

## Results Summary (Seed 42)
| Model | EER (%) | minDCF |
|-------|---------|--------|
| Baseline (Fixed $\lambda$) | 2.800 | 0.00550 |
| Improved (Dynamic $\lambda$) | 3.600 | 0.00690 |

**Note on Results:** While the dynamic weighting model achieved faster convergence during training, the fixed lambda baseline demonstrated slightly better generalization on the LibriSpeech test pairs. 
```

---
