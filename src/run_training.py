import torch
from models.smaLLM import smaLLM
from datasets.charDataset import charDataset
from training.training import train_model


# Main script to run training
if __name__ == "__main__":
    # ====== Dataset ======
    # Define your input data (replace with actual dataset for larger-scale training)
    data_path = "datasets/data.txt"
    with open(data_path, "r") as f:
        data = f.read()

    
    # Create a character-level dataset
    block_size = 128  # Length of input sequence
    dataset = charDataset(data=data, config=None, block_size=block_size)
    
    # ====== Model ======
    # Initialize the Transformer-based language model
    model = smaLLM(
        vocab_size=dataset.get_vocab_size(),
        embed_dim=128,      # Embedding dimension
        num_heads=4,        # Number of attention heads
        ff_dim=256,         # Feed-forward network dimension
        num_blocks=4,       # Number of Transformer blocks
        block_size=block_size,
        dropout=0.1         # Dropout probability
    )
    
    # ====== Device ======
    # Determine whether to use a GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ====== Training ======
    # Training configuration
    train_model(
        model=model,
        dataset=dataset,
        epochs=5,           # Number of epochs
        batch_size=32,      # Batch size
        lr=0.001,           # Initial learning rate
        device=device,
        max_grad_norm=1.0,  # Gradient clipping norm
        lr_decay_gamma=0.95 # Learning rate decay factor (Tried 0.95, 0.99 and 1.0)
    )
    
    # ====== Save the Model ======
    # Save the trained model to a file
    torch.save(model.state_dict(), "smaLLM_lr_decay_01.pth")
    print("\33[1mModel training complete. Model saved to 'smaLLM_lr_decay_01.pth'.\33[0m")

    print('_____\n\x1B[3m Dany A. Darghouth - December 2024')
    