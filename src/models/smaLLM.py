import torch
import torch.nn as nn
from models.transformerBlock import transformerBlock

class smaLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_blocks, block_size, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            transformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device).unsqueeze(0))
        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
    
    
def load_model(model_path, vocab_size, embed_dim, num_heads, ff_dim, num_blocks, block_size, dropout, device):
    """
    Load the trained model from a file.

    Args:
        model_path (str): Path to the saved model file.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimension of feed-forward layers.
        num_blocks (int): Number of Transformer blocks.
        block_size (int): Maximum sequence length.
        dropout (float): Dropout rate.
        device (torch.device): Device to load the model on.

    Returns:
        smaLLM: Loaded model.
    """
    # Initialize the model
    model = smaLLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        dropout=dropout
    )

    # Load the state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)  # Move to the specified device
    model.eval()  # Set to evaluation mode

    print(f"\x1B[3mModel loaded from {model_path}.")
    return model

