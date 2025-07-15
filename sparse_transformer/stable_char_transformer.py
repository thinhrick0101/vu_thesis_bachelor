import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import math
import random
import os
import gzip
import urllib.request
import wandb
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training
from torch.utils.checkpoint import checkpoint  # For gradient checkpointing
from tokenizers import Tokenizer  # For loading the BPE tokenizer
from sparse_attention import SparseMultiHeadAttention

def load_data(data_path, data_url=None):
    """
    Load text data from file or download if not available
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    # Download data if not available
    if not os.path.exists(data_path) and data_url:
        print(f"Downloading data from {data_url}")
        urllib.request.urlretrieve(data_url, data_path + '.gz')

        # Decompress .gz file
        with gzip.open(data_path + '.gz', 'rb') as f_in:
            with open(data_path, 'wb') as f_out:
                f_out.write(f_in.read())

    # Load data as raw bytes
    print(f"Loading data from {data_path}")
    with open(data_path, 'rb') as f:
        data = f.read()

    return data

class ByteTokenizer:
    """
    Byte-level tokenizer with a fixed vocabulary size of 256
    """
    def __init__(self):
        # Fixed vocabulary size of 256 (all possible bytes)
        self.vocab_size = 256
        
        # Create byte-to-index and index-to-byte mappings
        self.byte_to_idx = {i: i for i in range(256)}
        self.idx_to_byte = {i: i for i in range(256)}
        
    def encode(self, data):
        """Convert data to byte indices"""
        if isinstance(data, str):
            # Convert string to bytes
            data = data.encode('utf-8', errors='ignore')
        
        if isinstance(data, bytes):
            return torch.tensor([b for b in data], dtype=torch.long)
        elif isinstance(data, (list, tuple)):
            return torch.tensor([b % 256 for b in data], dtype=torch.long)
        elif torch.is_tensor(data):
            return data.long() % 256
        else:
            raise ValueError("Input must be string, bytes, list/tuple of integers, or tensor")
    
    def decode(self, indices):
        """Convert byte indices back to bytes/text"""
        if torch.is_tensor(indices):
            indices = indices.tolist()
        
        # Ensure all indices are within valid byte range
        bytes_data = bytes(i % 256 for i in indices)
        
        # Try to decode as UTF-8 first
        try:
            return bytes_data.decode('utf-8', errors='replace')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, return raw bytes for analysis
            return bytes_data

def create_batches(data, batch_size, seq_length):
    """
    Create batches of data for training

    Args:
        data: List of token indices
        batch_size: Number of sequences per batch
        seq_length: Length of each sequence

    Returns:
        List of (input, target) tuples
    """
    # Calculate the number of batches
    num_batches = (len(data) - 1) // (batch_size * seq_length)

    # Trim the data to fit into batches
    data = data[:num_batches * batch_size * seq_length + 1]

    # Convert data to tensor first
    data_tensor = torch.LongTensor(data)
    
    # Create input and target tensors using clone() and detach()
    x = data_tensor[:-1].clone().detach().view(batch_size, -1)
    y = data_tensor[1:].clone().detach().view(batch_size, -1)

    # Create batches
    batches = []
    for i in range(0, x.size(1), seq_length):
        # Get sequences of length seq_length
        if i + seq_length <= x.size(1):
            input_batch = x[:, i:i+seq_length].clone()
            target_batch = y[:, i:i+seq_length].clone()
            batches.append((input_batch, target_batch))

    return batches

class ImprovedPositionalEncoding(nn.Module):
    """
    Improved positional encoding with learnable parameters and better initialization
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(ImprovedPositionalEncoding, self).__init__()

        # Create base positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register buffer (persistent state)
        self.register_buffer('pe', pe)

        # Learnable scaling factor for positional encoding
        self.alpha = nn.Parameter(torch.ones(1))

        # Learnable position-wise feed-forward layer
        self.position_ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for the learnable components"""
        for module in self.position_ff.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, embedding_dim]

        Returns:
            Positional encoding added to input embeddings
        """
        # Get base positional encoding
        pos_enc = self.pe[:x.size(1), :].unsqueeze(0)

        # Apply learnable scaling
        pos_enc = self.alpha * pos_enc

        # Add positional encoding to input embeddings
        x = x + pos_enc

        # Apply position-wise feed-forward layer
        x = x + self.dropout(self.position_ff(self.norm(x)))

        return x

class EnhancedTransformerBlock(nn.Module):
    """
    Enhanced transformer encoder block with gradient checkpointing, improved attention,
    and better regularization techniques
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, activation_dropout=0.1, use_checkpoint=True,
                 attention_module=None):
        super(EnhancedTransformerBlock, self).__init__()

        # Multi-head self-attention with scaled dot-product attention
        self.self_attn = attention_module if attention_module is not None else nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=attention_dropout,  # Separate dropout for attention
            batch_first=True
        )

        # Improved feed-forward network with SwiGLU activation (better than GELU)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward * 2),  # Double size for SwiGLU
            nn.SiLU(),  # SiLU (Swish) activation
            nn.Dropout(activation_dropout),  # Separate dropout for activations
            nn.Linear(dim_feedforward * 2, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization with better epsilon
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Residual scaling factors (learnable)
        self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.ones(1))

        # Gradient checkpointing flag
        self.use_checkpoint = use_checkpoint

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with improved techniques"""
        # Initialize feed-forward network
        for module in self.feed_forward.modules():
            if isinstance(module, nn.Linear):
                # Use truncated normal distribution for better stability
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _attention_block(self, src, src_mask=None):
        """Self-attention block with pre-norm"""
        src2 = self.norm1(src)
        src2, attn_weights = self.self_attn(src2, src2, src2, attn_mask=src_mask, need_weights=True)
        return self.dropout1(self.gamma1 * src2)

    def _ff_block(self, src):
        """Feed-forward block with pre-norm"""
        src2 = self.norm2(src)
        src2 = self.feed_forward(src2)
        return self.dropout2(self.gamma2 * src2)

    def forward(self, src, src_mask=None):
        """
        Args:
            src: Input tensor of shape [batch_size, seq_length, embedding_dim]
            src_mask: Mask for self-attention

        Returns:
            Output tensor of shape [batch_size, seq_length, embedding_dim]
        """
        # Apply gradient checkpointing if enabled (saves memory during training)
        if self.use_checkpoint and self.training:
            # Self-attention block with gradient checkpointing
            src = src + checkpoint(
                lambda x, mask: self._attention_block(x, mask),
                src, src_mask,
                use_reentrant=False  # Add explicit use_reentrant parameter
            )

            # Feed-forward block with gradient checkpointing
            src = src + checkpoint(
                self._ff_block,
                src,
                use_reentrant=False  # Add explicit use_reentrant parameter
            )
        else:
            # Self-attention block (with pre-norm)
            src = src + self._attention_block(src, src_mask)

            # Feed-forward block (with pre-norm)
            src = src + self._ff_block(src)

        return src

class EnhancedCharTransformer(nn.Module):
    """
    Enhanced character-level transformer model for text generation
    with improved positional encoding, enhanced transformer blocks, token-level dropout,
    sophisticated initialization, and gradient checkpointing for memory efficiency
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward,
                 dropout=0.1, attention_dropout=0.1, activation_dropout=0.1,
                 token_dropout=0.05, use_checkpoint=True, stochastic_depth_prob=0.1,
                 attention_class=None, attention_kwargs=None):
        super(EnhancedCharTransformer, self).__init__()

        # Token embedding with weight tying preparation
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Embedding scale factor (optional but helps training)
        self.embed_scale = math.sqrt(d_model)

        # Improved positional encoding
        self.pos_encoder = ImprovedPositionalEncoding(d_model, dropout=dropout)

        # Enhanced transformer blocks with progressive dropout
        self.transformer_blocks = nn.ModuleList()
        for i in range(num_layers):
            # Gradually increase dropout in deeper layers
            layer_dropout = dropout * (1.0 + i * 0.1)
            layer_dropout = min(layer_dropout, 0.5)  # Cap at 0.5

            # Gradually increase attention dropout in deeper layers
            layer_attn_dropout = attention_dropout * (1.0 + i * 0.05)
            layer_attn_dropout = min(layer_attn_dropout, 0.4)  # Cap at 0.4

            # Create transformer block with custom attention if specified
            if attention_class is not None:
                attention = attention_class(
                    d_model,
                    nhead,
                    dropout=layer_attn_dropout,
                    **(attention_kwargs or {})
                )
            else:
                attention = nn.MultiheadAttention(
                    d_model,
                    nhead,
                    dropout=layer_attn_dropout,
                    batch_first=True
                )

            self.transformer_blocks.append(
                EnhancedTransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=layer_dropout,
                    attention_dropout=layer_attn_dropout,
                    activation_dropout=activation_dropout,
                    use_checkpoint=use_checkpoint,
                    attention_module=attention
                )
            )

        # Final layer normalization
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

        # Output projection (weight tied with embedding)
        self.output = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Token-level dropout for better generalization
        self.token_dropout = token_dropout

        # Stochastic depth probability
        self.stochastic_depth_prob = stochastic_depth_prob

        # Initialize parameters
        self._init_parameters()

        # Tie weights between embedding and output projection
        self.output.weight = self.embedding.weight

    def _init_parameters(self):
        """Initialize model parameters with improved techniques"""
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

        # Initialize output projection bias
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def _generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return ~mask  # Return boolean mask where True means masked positions

    def forward(self, src, mask=None):
        """
        Args:
            src: Input tensor of shape [batch_size, seq_length]
            mask: Optional mask for self-attention

        Returns:
            Output tensor of shape [batch_size, seq_length, vocab_size]
        """
        # Create causal mask if not provided
        if mask is None:
            mask = self._generate_square_subsequent_mask(src.size(1)).to(src.device)

        # Apply token-level dropout during training
        if self.training and self.token_dropout > 0:
            # Create a random mask for token dropout
            token_mask = torch.bernoulli(
                torch.full_like(src, 1 - self.token_dropout, dtype=torch.float)
            ).bool()

            # Replace dropped tokens with a special token (0 for simplicity)
            # This simulates missing or corrupted tokens
            src = torch.where(token_mask, src, torch.zeros_like(src))

        # Embed tokens and scale
        # [batch_size, seq_length] -> [batch_size, seq_length, d_model]
        x = self.embedding(src) * self.embed_scale

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply dropout
        x = self.dropout(x)

        # Pass through transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            # Apply stochastic depth (higher probability of skipping later layers)
            if self.training and i > 0:
                skip_prob = self.stochastic_depth_prob * (i / len(self.transformer_blocks))
                if random.random() < skip_prob:
                    continue

            x = block(x, src_mask=mask)

        # Apply final layer normalization
        x = self.norm(x)

        # Apply output projection
        output = self.output(x)

        return output

    def generate(self, prompt, max_length, temperature=0.7, top_k=20, top_p=0.9,
                repetition_penalty=1.2, tokenizer=None, device='cpu'):
        """
        Generate text from a prompt using byte-level sampling
        """
        self.eval()  # Set model to evaluation mode

        # Convert prompt to tensor if needed
        if isinstance(prompt, str) or isinstance(prompt, bytes):
            if tokenizer is None:
                raise ValueError("Tokenizer is required when prompt is a string or bytes")
            prompt_ids = tokenizer.encode(prompt)
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(device)
        else:
            prompt_tensor = prompt

        # Generate text
        generated = prompt_tensor.clone()

        # Keep track of past tokens for repetition penalty
        past_tokens = set()
        for token in generated[0].tolist():
            past_tokens.add(token)

        with torch.no_grad():
            for _ in range(max_length):
                try:
                    # Get predictions for the last token
                    # Use a sliding window approach for long sequences to save memory
                    if generated.size(1) > 1024:
                        # Use only the last 1024 tokens for context
                        context = generated[:, -1024:]
                    else:
                        context = generated

                    outputs = self(context)

                    # Apply temperature scaling with a safety check
                    next_token_logits = outputs[:, -1, :].clone()

                    # Ensure logits are valid for all 256 bytes
                    if next_token_logits.size(-1) != 256:
                        print(f"Warning: Expected 256 logits but got {next_token_logits.size(-1)}. Padding with -inf.")
                        padded_logits = torch.full((next_token_logits.size(0), 256), float('-inf'), device=device)
                        padded_logits[:, :next_token_logits.size(-1)] = next_token_logits
                        next_token_logits = padded_logits

                    # Check for NaN or infinite values
                    if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any():
                        print("Warning: NaN or infinite values detected in logits. Using uniform sampling.")
                        next_token = torch.randint(0, 256, (1, 1), device=device)
                    else:
                        # Apply repetition penalty
                        if repetition_penalty > 1.0:
                            for token_id in past_tokens:
                                if token_id < next_token_logits.size(-1):
                                    next_token_logits[:, token_id] /= repetition_penalty

                        # Apply temperature with a safety check
                        next_token_logits = next_token_logits / max(0.1, temperature)  # Prevent division by zero

                        # Apply top-k filtering
                        if top_k > 0:
                            top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                            min_value = top_k_values[:, -1].unsqueeze(-1)
                            next_token_logits = torch.where(
                                next_token_logits < min_value,
                                torch.ones_like(next_token_logits) * float('-inf'),
                                next_token_logits
                            )

                        # Apply top-p (nucleus) filtering with safety checks
                        if top_p < 1.0:
                            # Sort logits in descending order
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)

                            # Apply softmax with a safety check
                            sorted_probs = F.softmax(sorted_logits, dim=-1)

                            # Check for NaN values
                            if torch.isnan(sorted_probs).any():
                                print("Warning: NaN values detected in probabilities. Using uniform sampling.")
                                next_token = torch.randint(0, 256, (1, 1), device=device)
                                continue

                            # Calculate cumulative probabilities
                            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                            # Create mask for tokens to remove
                            sorted_indices_to_remove = cumulative_probs > top_p

                            # Keep at least one token
                            if sorted_indices_to_remove.all():
                                sorted_indices_to_remove[..., 0] = False

                            # Shift indices to keep the first token above threshold
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = False

                            # Apply the mask to the sorted indices
                            indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(
                                -1, sorted_indices, sorted_indices_to_remove
                            )

                            # Set removed indices to -inf
                            next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))

                        # Apply softmax to get probabilities with a safety check
                        probs = F.softmax(next_token_logits, dim=-1)

                        # Check for NaN values or invalid probabilities
                        if torch.isnan(probs).any() or (probs < 0).any() or (probs > 1).any():
                            print("Warning: Invalid probability values. Using uniform sampling.")
                            next_token = torch.randint(0, 256, (1, 1), device=device)
                        else:
                            # Sample from the distribution
                            next_token = torch.multinomial(probs, num_samples=1)

                    # Ensure the token is within valid byte range
                    next_token = next_token % 256

                    # Add the new token to past tokens for repetition penalty
                    past_tokens.add(next_token.item())

                    # Append the next token to the generated sequence
                    generated = torch.cat((generated, next_token), dim=1)

                except Exception as e:
                    print(f"Error during generation: {e}")
                    # Fall back to a safe token
                    next_token = torch.randint(0, 256, (1, 1), device=device)
                    generated = torch.cat((generated, next_token), dim=1)

        # Decode the generated text
        try:
            if tokenizer is not None:
                # First ensure all tokens are valid bytes
                valid_bytes = generated[0].clamp(0, 255)
                return tokenizer.decode(valid_bytes.tolist())
            else:
                return generated
        except Exception as e:
            print(f"Error during decoding: {e}")
            # Return raw bytes as a fallback
            return bytes(generated[0].clamp(0, 255).tolist())

class SparseTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward,
                 dropout=0.1, attention_dropout=0.1, activation_dropout=0.1,
                 token_dropout=0.05, use_checkpoint=True, stochastic_depth_prob=0.1,
                 attention_class=None, attention_kwargs=None):
        super(SparseTransformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.embed_scale = math.sqrt(d_model)
        
        self.pos_encoder = ImprovedPositionalEncoding(d_model, dropout=dropout)
        
        self.transformer_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            # Gradually increase dropout in deeper layers
            layer_dropout = dropout * (1.0 + i * 0.1)
            layer_dropout = min(layer_dropout, 0.5)  # Cap at 0.5

            # Gradually increase attention dropout in deeper layers
            layer_attn_dropout = attention_dropout * (1.0 + i * 0.05)
            layer_attn_dropout = min(layer_attn_dropout, 0.4)
            
            attention = SparseMultiHeadAttention(
                d_model,
                nhead,
                dropout=layer_attn_dropout,
                bias= True)
            
            self.transformer_blocks.append(
                EnhancedTransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=layer_dropout,
                    attention_dropout=layer_attn_dropout,
                    activation_dropout=activation_dropout,
                    use_checkpoint=use_checkpoint,
                    attention_module=attention
                )
            )
            self.norm = nn.LayerNorm(d_model, eps=1e-6)

        # Output projection (weight tied with embedding)
        self.output = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Token-level dropout for better generalization
        self.token_dropout = token_dropout

        # Stochastic depth probability
        self.stochastic_depth_prob = stochastic_depth_prob

        # Initialize parameters
        self._init_parameters()

        # Tie weights between embedding and output projection
        self.output.weight = self.embedding.weight

    def _init_parameters(self):
        """Initialize model parameters with improved techniques"""
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

        # Initialize output projection bias
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def _generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return ~mask  # Return boolean mask where True means masked positions

    def forward(self, src, mask=None):
        """
        Args:
            src: Input tensor of shape [batch_size, seq_length]
            mask: Optional mask for self-attention

        Returns:
            Output tensor of shape [batch_size, seq_length, vocab_size]
        """
        # Create causal mask if not provided
        if mask is None:
            mask = self._generate_square_subsequent_mask(src.size(1)).to(src.device)

        # Apply token-level dropout during training
        if self.training and self.token_dropout > 0:
            # Create a random mask for token dropout
            token_mask = torch.bernoulli(
                torch.full_like(src, 1 - self.token_dropout, dtype=torch.float)
            ).bool()

            # Replace dropped tokens with a special token (0 for simplicity)
            # This simulates missing or corrupted tokens
            src = torch.where(token_mask, src, torch.zeros_like(src))

        # Embed tokens and scale
        # [batch_size, seq_length] -> [batch_size, seq_length, d_model]
        x = self.embedding(src) * self.embed_scale

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply dropout
        x = self.dropout(x)

        # Pass through transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            # Apply stochastic depth (higher probability of skipping later layers)
            if self.training and i > 0:
                skip_prob = self.stochastic_depth_prob * (i / len(self.transformer_blocks))
                if random.random() < skip_prob:
                    continue

            x = block(x, src_mask=mask)

        # Apply final layer normalization
        x = self.norm(x)

        # Apply output projection
        output = self.output(x)

        return output

    def generate(self, prompt, max_length, temperature=0.7, top_k=20, top_p=0.9,
                repetition_penalty=1.2, tokenizer=None, device='cpu'):
        """
        Generate text from a prompt using byte-level sampling
        """
        self.eval()  # Set model to evaluation mode

        # Convert prompt to tensor if needed
        if isinstance(prompt, str) or isinstance(prompt, bytes):
            if tokenizer is None:
                raise ValueError("Tokenizer is required when prompt is a string or bytes")
            prompt_ids = tokenizer.encode(prompt)
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(device)
        else:
            prompt_tensor = prompt

        # Generate text
        generated = prompt_tensor.clone()

        # Keep track of past tokens for repetition penalty
        past_tokens = set()
        for token in generated[0].tolist():
            past_tokens.add(token)

        with torch.no_grad():
            for _ in range(max_length):
                try:
                    # Get predictions for the last token
                    # Use a sliding window approach for long sequences to save memory
                    if generated.size(1) > 1024:
                        # Use only the last 1024 tokens for context
                        context = generated[:, -1024:]
                    else:
                        context = generated

                    outputs = self(context)

                    # Apply temperature scaling with a safety check
                    next_token_logits = outputs[:, -1, :].clone()

                    # Ensure logits are valid for all 256 bytes
                    if next_token_logits.size(-1) != 256:
                        print(f"Warning: Expected 256 logits but got {next_token_logits.size(-1)}. Padding with -inf.")
                        padded_logits = torch.full((next_token_logits.size(0), 256), float('-inf'), device=device)
                        padded_logits[:, :next_token_logits.size(-1)] = next_token_logits
                        next_token_logits = padded_logits

                    # Check for NaN or infinite values
                    if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any():
                        print("Warning: NaN or infinite values detected in logits. Using uniform sampling.")
                        next_token = torch.randint(0, 256, (1, 1), device=device)
                    else:
                        # Apply repetition penalty
                        if repetition_penalty > 1.0:
                            for token_id in past_tokens:
                                if token_id < next_token_logits.size(-1):
                                    next_token_logits[:, token_id] /= repetition_penalty

                        # Apply temperature with a safety check
                        next_token_logits = next_token_logits / max(0.1, temperature)  # Prevent division by zero

                        # Apply top-k filtering
                        if top_k > 0:
                            top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                            min_value = top_k_values[:, -1].unsqueeze(-1)
                            next_token_logits = torch.where(
                                next_token_logits < min_value,
                                torch.ones_like(next_token_logits) * float('-inf'),
                                next_token_logits
                            )

                        # Apply top-p (nucleus) filtering with safety checks
                        if top_p < 1.0:
                            # Sort logits in descending order
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)

                            # Apply softmax with a safety check
                            sorted_probs = F.softmax(sorted_logits, dim=-1)

                            # Check for NaN values
                            if torch.isnan(sorted_probs).any():
                                print("Warning: NaN values detected in probabilities. Using uniform sampling.")
                                next_token = torch.randint(0, 256, (1, 1), device=device)
                                continue

                            # Calculate cumulative probabilities
                            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                            # Create mask for tokens to remove
                            sorted_indices_to_remove = cumulative_probs > top_p

                            # Keep at least one token
                            if sorted_indices_to_remove.all():
                                sorted_indices_to_remove[..., 0] = False

                            # Shift indices to keep the first token above threshold
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = False

                            # Apply the mask to the sorted indices
                            indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(
                                -1, sorted_indices, sorted_indices_to_remove
                            )

                            # Set removed indices to -inf
                            next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))

                        # Apply softmax to get probabilities with a safety check
                        probs = F.softmax(next_token_logits, dim=-1)

                        # Check for NaN values or invalid probabilities
                        if torch.isnan(probs).any() or (probs < 0).any() or (probs > 1).any():
                            print("Warning: Invalid probability values. Using uniform sampling.")
                            next_token = torch.randint(0, 256, (1, 1), device=device)
                        else:
                            # Sample from the distribution
                            next_token = torch.multinomial(probs, num_samples=1)

                    # Ensure the token is within valid byte range
                    next_token = next_token % 256

                    # Add the new token to past tokens for repetition penalty
                    past_tokens.add(next_token.item())

                    # Append the next token to the generated sequence
                    generated = torch.cat((generated, next_token), dim=1)

                except Exception as e:
                    print(f"Error during generation: {e}")
                    # Fall back to a safe token
                    next_token = torch.randint(0, 256, (1, 1), device=device)
                    generated = torch.cat((generated, next_token), dim=1)

        # Decode the generated text
        try:
            if tokenizer is not None:
                # First ensure all tokens are valid bytes
                valid_bytes = generated[0].clamp(0, 255)
                return tokenizer.decode(valid_bytes.tolist())
            else:
                return generated
        except Exception as e:
            print(f"Error during decoding: {e}")
            # Return raw bytes as a fallback
            return bytes(generated[0].clamp(0, 255).tolist())
        

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0.0, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (1.0 - min_lr) * cosine_decay

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def train_model(model, train_batches, val_batches=None, num_epochs=5, learning_rate=0.0001,
                weight_decay=0.01, warmup_steps=0, min_lr=0.0, device=None, patience=3, 
                label_smoothing=0.0, gradient_accumulation_steps=1, use_mixed_precision=True,
                use_cosine_schedule=False, use_wandb=False):
    """
    Train the model and evaluate on validation set with mixed precision and gradient accumulation

    Args:
        model: The model to train
        train_batches: List of training batches
        val_batches: List of validation batches (optional)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for regularization
        warmup_steps: Number of warmup steps for learning rate scheduling
        min_lr: Minimum learning rate for the optimizer
        device: Device to use for training
        patience: Number of epochs to wait for improvement before early stopping
        label_smoothing: Label smoothing factor for regularization
        gradient_accumulation_steps: Number of steps to accumulate gradients before updating weights
        use_mixed_precision: Whether to use mixed precision training (FP16)
        use_cosine_schedule: Whether to use cosine learning rate schedule
        use_wandb: Whether to log metrics to Weights & Biases

    Returns:
        Trained model and training metrics
    """
    # Determine device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    model = model.to(device)
    print(f"Training on device: {device}")

    if use_wandb:
        wandb.watch(model, log='all', log_freq=100)

    # Initialize mixed precision training if available and requested
    use_amp = use_mixed_precision and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None  # Remove device_type parameter
    if use_amp:
        print("Using mixed precision training (FP16)")

    # Print gradient accumulation info
    if gradient_accumulation_steps > 1:
        print(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
        effective_batch_size = train_batches[0][0].size(0) * gradient_accumulation_steps
        print(f"Effective batch size: {effective_batch_size}")

    # Define loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Define optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Calculate total number of training steps
    total_steps = len(train_batches) * num_epochs // gradient_accumulation_steps

    # Create learning rate scheduler
    if warmup_steps > 0:
        if use_cosine_schedule:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                min_lr=min_lr
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=min_lr
        )

    # For tracking metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    no_improvement_count = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()

        # Zero the gradients at the beginning of each epoch
        optimizer.zero_grad()

        # Process each batch
        for batch_idx, (inputs, targets) in enumerate(train_batches):
            batch_start_time = time.time()
            # Move tensors to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass with mixed precision if enabled
            if use_amp:
                with autocast():
                    # Forward pass
                    forward_pass_start_time = time.time()
                    outputs = model(inputs)
                    forward_pass_end_time = time.time()

                    # Reshape for loss calculation
                    outputs = outputs.reshape(-1, outputs.size(-1))
                    targets = targets.reshape(-1)

                    # Calculate loss
                    loss = criterion(outputs, targets)
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Update weights if we've accumulated enough gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_batches):
                    # Unscale gradients for clipping
                    scaler.unscale_(optimizer)

                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Update weights with gradient scaling
                    scaler.step(optimizer)
                    scaler.update()

                    # Update learning rate AFTER optimizer step for per-step schedulers
                    if warmup_steps > 0: # Check if it's a per-step scheduler
                        scheduler.step()
                    
                    optimizer.zero_grad() # Zero gradients after update and scheduler step

            else:
                # Standard precision training
                # Forward pass
                forward_pass_start_time = time.time()
                outputs = model(inputs)
                forward_pass_end_time = time.time()

                # Reshape for loss calculation
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)

                # Calculate loss
                loss = criterion(outputs, targets)
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Update weights if we've accumulated enough gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_batches):
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Update weights
                    optimizer.step()
                    # Update learning rate AFTER optimizer step for per-step schedulers
                    if warmup_steps > 0: # Check if it's a per-step scheduler
                        scheduler.step()

                    # Zero gradients after update and scheduler step
                    optimizer.zero_grad()

            # Track loss (use the unscaled loss for logging)
            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1

            # Print batch progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                current_batch_loss = loss.item() * gradient_accumulation_steps
                if torch.isnan(torch.tensor(current_batch_loss)) or torch.isinf(torch.tensor(current_batch_loss)):
                    print(f"ERROR: NaN or Inf detected in current_batch_loss: {current_batch_loss}")

                # --- Calculate and log metrics ---
                forward_pass_latency = (forward_pass_end_time - forward_pass_start_time) * 1000 # in ms
                batch_time = time.time() - batch_start_time
                num_samples = inputs.size(0) # batch_size
                throughput = num_samples / batch_time if batch_time > 0 else 0
                
                try:
                    batch_perplexity = math.exp(current_batch_loss)
                except (OverflowError, ValueError):
                    batch_perplexity = float('inf')

                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_batches)}, "
                      f"Loss: {current_batch_loss:.4f}, PPL: {batch_perplexity:.2f}")
                if use_wandb:
                    wandb.log({
                        "batch_loss": current_batch_loss,
                        "lr": optimizer.param_groups[0]['lr'],
                        "batch_perplexity": batch_perplexity,
                        "forward_pass_latency_ms": forward_pass_latency,
                        "throughput_samples_per_sec": throughput
                    })

        # Calculate average loss for the epoch
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)

        # Calculate perplexity
        print(f"DEBUG: avg_loss before math.exp: {avg_loss}") # Added debug print
        if torch.isnan(torch.tensor(avg_loss)) or torch.isinf(torch.tensor(avg_loss)) or avg_loss > 700: # Prevent overflow
            print(f"Warning: avg_loss is problematic ({avg_loss}), perplexity will be very high or error.")
            perplexity = float('inf') # Assign inf if loss is too high for math.exp
        else:
            perplexity = math.exp(avg_loss)

        # Evaluate on validation set if provided
        if val_batches is not None:
            model.eval()
            val_total_loss = 0
            val_num_batches = 0

            with torch.no_grad():
                for inputs, targets in val_batches:
                    # Move tensors to device
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # Forward pass (use autocast for consistency if mixed precision is enabled)
                    if use_amp:
                        with autocast():
                            outputs = model(inputs)

                            # Reshape for loss calculation
                            outputs = outputs.reshape(-1, outputs.size(-1))
                            targets = targets.reshape(-1)

                            # Calculate loss
                            loss = criterion(outputs, targets)
                    else:
                        outputs = model(inputs)

                        # Reshape for loss calculation
                        outputs = outputs.reshape(-1, outputs.size(-1))
                        targets = targets.reshape(-1)

                        # Calculate loss
                        loss = criterion(outputs, targets)

                    val_total_loss += loss.item()
                    val_num_batches += 1

            # Calculate average validation loss
            val_avg_loss = val_total_loss / val_num_batches
            val_losses.append(val_avg_loss)
            
            # Calculate validation perplexity
            if torch.isnan(torch.tensor(val_avg_loss)) or torch.isinf(torch.tensor(val_avg_loss)) or val_avg_loss > 700:
                val_perplexity = float('inf')
            else:
                val_perplexity = math.exp(val_avg_loss)

            # Print epoch summary
            end_time = time.time()
            epoch_duration = end_time - start_time
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Train PPL: {perplexity:.2f} | "
                  f"Val Loss: {val_avg_loss:.4f} | Val PPL: {val_perplexity:.2f} | Time: {epoch_duration:.2f}s")
            
            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "train_perplexity": perplexity,
                    "val_loss": val_avg_loss,
                    "val_perplexity": val_perplexity
                })

            # Check for early stopping
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                best_model_state = model.state_dict().copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping if no improvement
            if no_improvement_count >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                break
        else:
            # Print epoch summary without validation
            end_time = time.time()
            epoch_duration = end_time - start_time
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Train PPL: {perplexity:.2f} | "
                  f"Time: {epoch_duration:.2f}s")
            
            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "train_perplexity": perplexity
                })

    # Load best model state if early stopping was used
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model state from early stopping.")

    # Return trained model and history
    if val_losses:
        return model, (train_losses, val_losses)
    else:
        return model, (train_losses, [])

def visualize_results(train_losses, val_losses=None, filename='enhanced_char_transformer_loss.png'):
    """
    Visualize training and validation losses

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        filename: Name of the output file
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss', marker='o', markersize=4, linestyle='-', linewidth=1)

    if val_losses:
        plt.plot(val_losses, label='Validation Loss', marker='s', markersize=4, linestyle='-', linewidth=1)

        # Plot the best validation loss
        best_epoch = val_losses.index(min(val_losses))
        best_loss = val_losses[best_epoch]
        plt.plot(best_epoch, best_loss, 'r*', markersize=10, label=f'Best Val Loss: {best_loss:.4f}')

    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add perplexity as secondary y-axis
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Create perplexity ticks based on loss values
    loss_ticks = ax1.get_yticks()
    perplexity_ticks = [math.exp(x) for x in loss_ticks if x > 0]
    ax2.set_yticks(perplexity_ticks)
    ax2.set_yticklabels([f'{x:.1f}' for x in perplexity_ticks])
    ax2.set_ylabel('Perplexity', fontsize=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    # Clear CUDA cache at the start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Data parameters
    data_path = 'data/enwik8'
    data_url = 'https://codeberg.org/pbm/former/raw/branch/master/data/enwik8.gz'

    # Model hyperparameters - optimized for byte-level
    d_model = 512
    nhead = 8
    num_layers = 12
    dim_feedforward = 2048
    dropout = 0.1
    attention_dropout = 0.1
    activation_dropout = 0.1
    token_dropout = 0.05
    use_checkpoint = True
    stochastic_depth_prob = 0.1

    # Training hyperparameters - optimized for byte-level
    batch_size = 32
    seq_length = 1024  # Good sequence length for byte-level
    num_epochs = 100
    learning_rate = 1e-4
    min_lr = 1e-5
    weight_decay = 0.1
    label_smoothing = 0.0
    gradient_accumulation_steps = 8
    use_mixed_precision = True
    warmup_epochs = 5

    # Set memory allocation settings
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    # Load data as raw bytes
    print("Loading data...")
    data = load_data(data_path, data_url)
    print(f"Data loaded: {len(data)} bytes")

    # Create byte tokenizer
    print("Creating byte tokenizer...")
    tokenizer = ByteTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size} bytes (fixed)")

    # Encode the data
    print("Encoding data...")
    encoded_data = tokenizer.encode(data)
    print(f"Encoded length: {len(encoded_data)} tokens")

    # Analyze byte distribution
    print("\nAnalyzing byte distribution...")
    byte_counts = torch.bincount(encoded_data, minlength=256)
    print(f"Number of unique bytes: {(byte_counts > 0).sum().item()}")
    print(f"Min byte value: {encoded_data.min().item()}")
    print(f"Max byte value: {encoded_data.max().item()}")
    print("\nMost common bytes:")
    top_bytes = torch.topk(byte_counts, k=10)
    for value, count in zip(top_bytes.indices.tolist(), top_bytes.values.tolist()):
        char_repr = chr(value) if 32 <= value <= 126 else f"<byte {value}>"
        print(f"Byte {value} ({char_repr}): {count:,} occurrences")

    # Split data into training and validation sets (90% / 10%)
    split_idx = int(len(encoded_data) * 0.9)
    train_data = encoded_data[:split_idx]
    val_data = encoded_data[split_idx:]

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create batches
    print("Creating batches...")
    train_batches = create_batches(train_data, batch_size, seq_length)
    val_batches = create_batches(val_data, batch_size, seq_length)
    print(f"Created {len(train_batches)} training batches and {len(val_batches)} validation batches")

    # Calculate warmup steps after creating batches
    warmup_steps = len(train_batches) * warmup_epochs

    # Create enhanced model with stochastic depth
    model = EnhancedCharTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        attention_dropout=attention_dropout,
        activation_dropout=activation_dropout,
        token_dropout=token_dropout,
        use_checkpoint=use_checkpoint,
        stochastic_depth_prob=stochastic_depth_prob
    )

    # Move model to device
    model = model.to(device)

    # Print model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {trainable_params:,} trainable out of {total_params:,} total")

    # Train model with enhanced settings
    print("\n=== Training Enhanced Character Transformer Model ===")
    model, (train_losses, val_losses) = train_model(
        model=model,
        train_batches=train_batches,
        val_batches=val_batches,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        min_lr=min_lr,  # Added parameter
        device=device,
        patience=8,  # Increased patience
        label_smoothing=label_smoothing,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_mixed_precision=use_mixed_precision,
        use_cosine_schedule=True,  # Added cosine schedule
        use_wandb=True  # Added wandb logging
    )

    # Visualize results
    visualize_results(train_losses, val_losses, 'enhanced_char_transformer_loss.png')
    print("\nTraining visualization saved to enhanced_char_transformer_loss.png")

    # Generate some text
    print("\n=== Generating Text ===")
    prompt = "The quick brown fox"
    generated_text = model.generate(
        prompt=prompt,
        max_length=500,
        temperature=0.6,  # Lower temperature for more coherent text
        top_k=5,  # Increased top_k for more diversity
        top_p=0.95,  # Balanced top_p for coherence and diversity
        repetition_penalty=1.2,  # Add repetition penalty to avoid loops
        tokenizer=tokenizer,
        device=device
    )
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

    # Save model
    torch.save(model.state_dict(), 'enhanced_char_transformer_model.pt')
    print("Model saved to enhanced_char_transformer_model.pt")

    # Try generating with different temperatures
    print("\n=== Generating with Different Temperatures ===")
    for temp in [0.5, 0.7, 0.9]:
        generated_text = model.generate(
            prompt=prompt,
            max_length=500,
            temperature=temp,
            top_k=50,
            top_p=0.92,
            repetition_penalty=1.2,
            tokenizer=tokenizer,
            device=device
        )
        print(f"\nTemperature: {temp}")
        print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()
