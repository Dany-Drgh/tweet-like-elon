import torch
import torch.nn as nn

def train_model(model, dataset, epochs, batch_size, lr, device, max_grad_norm=1.0, lr_decay_gamma=0.95):
    """
    Train a PyTorch model with learning rate decay and gradient clipping.

    Args:
        model (nn.Module): The model to train.
        dataset (CharDataset): The character-level dataset for training.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Initial learning rate.
        device (torch.device): The device (CPU or GPU) to train on.
        max_grad_norm (float): Maximum gradient norm for clipping.
        lr_decay_step (int): Step size for learning rate decay.
        lr_decay_gamma (float): Decay factor for learning rate.
    """
    # ====== DataLoader ======
    # Create a DataLoader to handle batching and shuffling of the dataset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # ====== Optimizer and Loss Function ======
    # Adam optimizer for training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Cross-entropy loss for sequence prediction
    criterion = nn.CrossEntropyLoss()

    # ====== Learning Rate Decay ======
    # Exponential learning rate decay
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_gamma)

    model.to(device)

    # ====== Training Loop ======
    print("\33[1mStarting Training...\33[0m\n")

    model.train()  # Set the model to training mode

    for epoch in range(epochs):
        total_loss = 0  # Track total loss for the epoch
        
        # Iterate through the dataset in batches
        for input_seq, target_seq in data_loader:
            # Move input and target sequences to the same device as the model
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            # ====== Forward Pass ======
            logits = model(input_seq)  # Model outputs logits of shape (B, T, vocab_size)

            # ====== Loss Calculation ======
            # Reshape logits and target for cross-entropy loss
            loss = criterion(logits.view(-1, logits.size(-1)), target_seq.view(-1))
            
            # ====== Backward Pass ======
            optimizer.zero_grad()
            loss.backward()  # Compute gradients

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # Gradient clipping

            optimizer.step()  # Optimazation step

            total_loss += loss.item()  # Accumulate the loss for the batch

        scheduler.step()  # Update the learning rate for learning rate decay
    
        # ====== Epoch Summary ======
        avg_loss = total_loss / len(data_loader)  # Calculate average loss
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")