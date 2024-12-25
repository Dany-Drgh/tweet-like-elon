import torch
import torch.nn as nn

class transformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        """
        Args:
            embed_dim: Int
                Dimension of the input embeddings
            num_heads: Int
                Number of attention heads
            ff_dim: Int
                Dimension of the feed-forward network
            dropout: Float
                Dropout rate
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)  # LayerNorm for stable training
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.ln2 = nn.LayerNorm(embed_dim)  # Another LayerNorm before the feed-forward

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),  # First feed-forward layer
            nn.ReLU(),                     # Activation function
            nn.Linear(ff_dim, embed_dim)   # Map back to original size
        )

        self.dropout = nn.Dropout(dropout)  # Regularization

    def forward(self, x):
        # Self-Attention with residual connection
        attn_out, _ = self.attn(x, x, x, attn_mask=self._causal_mask(x))  # Query, Key, Value are all `x`
        x = x + self.dropout(attn_out)  # Residual connection
        x = self.ln1(x)  # Normalize the result

        # Feed-Forward with residual connection
        ff_out = self.ff(x)  # Apply feed-forward network
        x = x + self.dropout(ff_out)  # Residual connection
        x = self.ln2(x)  # Normalize the result again

        return x
    
    def _causal_mask(self, x):
        # Generate a causal mask to ensure attention looks only at past tokens
        seq_len = x.size(1)
        mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1)
        return mask.bool()