import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. Define the Fairness Loss Module ---
class FairASRLoss(nn.Module):
    def __init__(self, lambda_fairness=0.5):
        super(FairASRLoss, self).__init__()
        # Standard ASR loss (Using MSE here as a proxy for CTC Loss for simplicity)
        self.standard_criterion = nn.MSELoss(reduction='none') 
        self.lambda_fairness = lambda_fairness

    def forward(self, predictions, targets, demographics):
        """
        predictions: Model outputs
        targets: Ground truth labels
        demographics: Tensor of demographic group IDs (e.g., 0=Majority, 1=Minority)
        """
        # 1. Calculate standard base loss for every item in the batch
        base_losses = self.standard_criterion(predictions, targets)
        
        # 2. Separate losses by demographic group
        # Assuming group 0 is majority (e.g., young/female) and 1 is minority (e.g., old/male)
        mask_majority = (demographics == 0)
        mask_minority = (demographics == 1)
        
        loss_majority = base_losses[mask_majority].mean() if mask_majority.sum() > 0 else torch.tensor(0.0)
        loss_minority = base_losses[mask_minority].mean() if mask_minority.sum() > 0 else torch.tensor(0.0)
        
        # Total standard loss (average over whole batch)
        standard_loss = base_losses.mean()
        
        # 3. Calculate the Fairness Penalty (Absolute gap between groups)
        fairness_penalty = torch.abs(loss_majority - loss_minority)
        
        # 4. Combine for final loss
        total_loss = standard_loss + (self.lambda_fairness * fairness_penalty)
        
        return total_loss, standard_loss, fairness_penalty

# --- 2. Create a Mock ASR Model and Training Loop ---
def run_training_simulation():
    print("--- STARTING FAIRNESS TRAINING SIMULATION ---")
    
    # Mock parameters
    batch_size = 32
    epochs = 5
    
    # A tiny mock acoustic model
    model = nn.Sequential(
        nn.Linear(80, 128),
        nn.ReLU(),
        nn.Linear(128, 40) # Outputting vocabulary logits
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = FairASRLoss(lambda_fairness=0.8) # Strong fairness enforcement
    
    for epoch in range(epochs):
        model.train()
        
        # -- SIMULATE A BATCH OF DATA --
        # 80-dimensional Mel-Spectrogram features
        inputs = torch.randn(batch_size, 80) 
        # Target transcriptions (mocked)
        targets = torch.randn(batch_size, 40) 
        
        # Simulate demographics based on our audit! 
        # ~80% Majority (Group 0), ~20% Minority (Group 1)
        demographics = torch.bernoulli(torch.full((batch_size,), 0.2)).long()
        
        # -- STANDARD PYTORCH TRAINING STEP --
        optimizer.zero_grad()
        
        predictions = model(inputs)
        
        # Calculate our custom Fair Loss
        loss, std_loss, fair_penalty = criterion(predictions, targets, demographics)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}] | Total Loss: {loss.item():.4f} "
              f"(ASR Loss: {std_loss.item():.4f}, Fairness Gap Penalty: {fair_penalty.item():.4f})")

    print("--- TRAINING COMPLETE ---")
    print("The model has updated its weights to balance accuracy across demographic groups.")

if __name__ == "__main__":
    run_training_simulation()