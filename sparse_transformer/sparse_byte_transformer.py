"""
Byte-level transformer model with sparse attention patterns for Enwik8 dataset.
This is a direct replacement for EnhancedCharTransformer, implementing static sparse attention
patterns derived from cluster analysis of dense attention heads.
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from sparse_attention import SparseTransformerLayer


class ImprovedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(ImprovedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # changed from pe.unsqueeze(0).transpose(0,1) to be [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [B, L, E]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SparseByteTransformer(nn.Module):
    """
    Byte-level language model using sparse transformer for Enwik8 dataset.
    This is a direct replacement for EnhancedCharTransformer, using sparse attention.
    
    The model uses:
    - Fixed vocabulary size of 256 (byte-level)
    - Scaled token embeddings
    - Improved positional encoding
    - Token-level dropout
    - Stack of sparse transformer layers
    - Weight-tied output projection
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size # Should be 256 for byte-level
        self.d_model = config.d_model

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoder = ImprovedPositionalEncoding(self.d_model, self.config.dropout, self.config.seq_length)
        
        # Transformer Encoder Layers
        mask_subset = getattr(self.config, 'mask_subset', '0123')  # Default to all clusters
        self.transformer_encoder = nn.ModuleList([
            SparseTransformerLayer(
                d_model=self.d_model,
                nhead=self.config.nhead,
                dim_feedforward=self.config.dim_feedforward,
                dropout=self.config.dropout,
                mask_subset=mask_subset
            ) for _ in range(self.config.num_layers)
        ])

        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        
        # Weight tying
        if getattr(self.config, 'tie_weights', True):
            self.output_projection.weight = self.token_embedding.weight

        self.embed_scale = math.sqrt(self.d_model) if getattr(self.config, 'scale_embeddings', True) else 1.0
        self.token_dropout_p = getattr(self.config, 'token_dropout', 0.0)

    def forward(self, src_tokens, padding_mask=None):
        # src_tokens: [B, L] (integer token ids)
        # padding_mask: [B, L] (boolean, True if NOT padded, False if padded) - OPTIONAL

        # Embedding and positional encoding
        x = self.token_embedding(src_tokens) * self.embed_scale # [B, L, E]
        
        if self.training and self.token_dropout_p > 0:
            # Apply token dropout (set entire embedding vector to zero)
            mask = torch.rand_like(x[:,:,0]) < self.token_dropout_p # [B,L]
            x[mask] = 0.0

        x = self.pos_encoder(x) # [B, L, E]

        # Prepare src_key_padding_mask for SparseTransformerLayer
        # It expects True for PADDED tokens.
        # Our input `padding_mask` is True for NON-PADDED tokens.
        src_key_padding_mask_for_layers = None
        if padding_mask is not None:
            src_key_padding_mask_for_layers = ~padding_mask # Invert: True means PADDED

        # Transformer layers
        for layer in self.transformer_encoder:
            # Layers no longer take attn_mask for causality
            x = layer(x, src_key_padding_mask=src_key_padding_mask_for_layers) 
            # x shape: [B, L, E]

        # Output projection
        logits = self.output_projection(x) # [B, L, vocab_size]
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def generate(self, prefix_tokens, max_new_tokens, temperature=1.0):
        """
        Generate new tokens autoregressively given a prefix.
        
        Args:
            prefix_tokens: Starting sequence [batch_size, prefix_length]
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (1.0 = standard, <1.0 = more focused)
        
        Returns:
            tokens: Generated sequence [batch_size, prefix_length + max_new_tokens]
        """
        self.eval()
        with torch.no_grad():
            # Start with the prefix
            tokens = prefix_tokens.clone()
            
            # Generate new tokens one at a time
            for _ in range(max_new_tokens):
                # Get predictions
                logits = self(tokens)  # [B, L, 256]
                
                # Only take the last token's predictions
                next_token_logits = logits[:, -1, :] / temperature  # [B, 256]
                
                # Sample from the distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
                
                # Append to the sequence
                tokens = torch.cat([tokens, next_token], dim=1)
            
            return tokens