import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio

try:
    torchaudio.list_audio_backends()
except AttributeError:
    torchaudio.list_audio_backends = lambda: ["soundfile", "sox_io"]

try:
    from torchaudio import _torchcodec
    import torchaudio.backend.sox_io_backend as sox
    torchaudio.load = sox.load
except ImportError:
    pass

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from speechbrain.inference.speaker import EncoderClassifier
from dataset import LibriSpeechTripletDataset
from models import DisentanglingAutoEncoder, Discriminator, GRL, MAPCLoss, DynamicLossWeighter

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_loss_curves(loss_history, config_name):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, marker='o', linestyle='-')
    plt.title(f"Total Loss Curve - {config_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    
    save_path = f"results/loss_curve_{config_name}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")

def train(config_path, seed):
    set_seed(seed)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{config['experiment_name']}_seed{seed}"
    print(f"Running: {run_name} on {device}")
    
    os.makedirs('results', exist_ok=True)
    
    dataset = LibriSpeechTripletDataset()
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    
    print("Loading pre-trained ECAPA-TDNN...")
    ecapa_classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    for param in ecapa_classifier.mods.parameters():
        param.requires_grad = False
    ecapa_classifier.eval()

    num_libri_speakers = len(dataset.valid_speakers)
    autoencoder = DisentanglingAutoEncoder().to(device)
    disc_S = Discriminator(is_speaker=True, num_speakers=num_libri_speakers).to(device)
    disc_E = Discriminator(is_speaker=False).to(device)
    disc_adv_E = Discriminator(is_speaker=False).to(device)
    grl = GRL(alpha=1.0)
    
    crit_l1 = nn.L1Loss()
    crit_ce = nn.CrossEntropyLoss()
    crit_triplet = nn.TripletMarginLoss(margin=1.0)
    crit_mapc = MAPCLoss()
    
    dynamic_weighter = DynamicLossWeighter().to(device) if config['use_dynamic_loss'] else None
    
    params = list(autoencoder.parameters()) + list(disc_S.parameters()) + \
             list(disc_E.parameters()) + list(disc_adv_E.parameters())
    if dynamic_weighter:
        params += list(dynamic_weighter.parameters())
        
    optimizer = optim.Adam(params, lr=config['learning_rate'])
    
    autoencoder.train()
    loss_history = [] 
    
    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        for x_1, x_2, x_3, labels in loader:
            x_1, x_2, x_3, labels = x_1.to(device), x_2.to(device), x_3.to(device), labels.to(device)
            
            with torch.no_grad():
                emb_1 = ecapa_classifier.encode_batch(x_1).squeeze(1)
                emb_2 = ecapa_classifier.encode_batch(x_2).squeeze(1)
                emb_3 = ecapa_classifier.encode_batch(x_3).squeeze(1)
            
            e_spk_1, e_env_1, recon_1 = autoencoder(emb_1)
            e_spk_2, e_env_2, recon_2 = autoencoder(emb_2, swap_codes=True, x_swap=emb_3)
            e_spk_3, e_env_3, recon_3 = autoencoder(emb_3, swap_codes=True, x_swap=emb_2)

            l_recons = crit_l1(recon_1, emb_1) + crit_l1(recon_2, emb_2) + crit_l1(recon_3, emb_3)
            l_spk = crit_ce(disc_S(e_spk_1), labels)
            l_env_env = crit_triplet(disc_E(e_env_1), disc_E(e_env_2), disc_E(e_env_3))
            l_env_spk = crit_triplet(disc_adv_E(grl(e_spk_1)), disc_adv_E(grl(e_spk_2)), disc_adv_E(grl(e_spk_3)))
            l_corr = crit_mapc(e_spk_1, e_env_1)
            
            if config['use_dynamic_loss']:
                loss = dynamic_weighter([l_spk, l_recons, l_env_env, l_env_spk, l_corr])
            else:
                lbd = config['lambdas']
                loss = (lbd['spk'] * l_spk) + (lbd['recons'] * l_recons) + \
                       (lbd['env_env'] * l_env_env) + (lbd['env_spk'] * l_env_spk) + \
                       (lbd['corr'] * l_corr)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_epoch_loss = epoch_loss / len(loader)
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {avg_epoch_loss:.4f}")

    model_save_path = f"results/best_model_{run_name}.pt"
    torch.save(autoencoder.state_dict(), model_save_path)
    plot_loss_curves(loss_history, run_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.config, args.seed)