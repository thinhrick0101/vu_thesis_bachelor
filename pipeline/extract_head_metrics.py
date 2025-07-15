#!/usr/bin/env python3
"""
Generates head_metrics.csv with columns: layer, head, entropy, sparsity, distance
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import sys
import os
from pathlib import Path
import random

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from dense_transformer.stable_char_transformer import EnhancedCharTransformer
except ImportError:
    print("Could not import EnhancedCharTransformer. Make sure stable_char_transformer.py is available.")
    sys.exit(1)

class AttentionMetricsExtractor:
    def __init__(self, model):
        self.model = model

    def extract_metrics(self, input_seqs):
        """
        Extract attention metrics from real model attention for a batch of input sequences.
        input_seqs: Tensor of shape [num_seqs, seq_len]
        Returns: list of dicts with metrics for each (layer, head)
        """
        device = next(self.model.parameters()).device
        num_layers = len(self.model.transformer_blocks)
        num_heads = self.model.transformer_blocks[0].self_attn.num_heads
        seq_len = input_seqs.size(1)
        num_seqs = input_seqs.size(0)

        # Storage for all attention weights: [num_layers][num_seqs, num_heads, seq_len, seq_len]
        all_attn = [[] for _ in range(num_layers)]

        # Hook function to capture attention weights
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output: (attn_output, attn_weights)
                attn_weights = output[1]  # [batch, num_heads, seq_len, seq_len]
                all_attn[layer_idx].append(attn_weights.detach().cpu())
            return hook_fn

        # Register hooks
        hooks = []
        for i, block in enumerate(self.model.transformer_blocks):
            h = block.self_attn.register_forward_hook(make_hook(i))
            hooks.append(h)

        # Run all sequences through the model
        with torch.no_grad():
            _ = self.model(input_seqs.to(device))

        # Remove hooks
        for h in hooks:
            h.remove()

        # Stack attention weights for each layer: [num_seqs, num_heads, seq_len, seq_len]
        all_attn = [torch.cat(layer_attn, dim=0) for layer_attn in all_attn]
        # all_attn: list of [num_seqs, num_heads, seq_len, seq_len] for each layer

        # Compute metrics for each (layer, head), averaged over all sequences
        metrics_data = []
        for layer_idx, layer_attn in enumerate(all_attn):
            # layer_attn: [num_seqs, num_heads, seq_len, seq_len]
            for head_idx in range(num_heads):
                # Collect metrics for each sequence, then average
                entropy_list = []
                sparsity_list = []
                distance_list = []
                for seq_idx in range(num_seqs):
                    attn_matrix = layer_attn[seq_idx, head_idx]  # [seq_len, seq_len]
                    metrics = self._calculate_head_metrics(attn_matrix)
                    entropy_list.append(metrics['entropy'])
                    sparsity_list.append(metrics['sparsity'])
                    distance_list.append(metrics['distance'])
                metrics_data.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'entropy': float(np.mean(entropy_list)),
                    'sparsity': float(np.mean(sparsity_list)),
                    'distance': float(np.mean(distance_list)),
                })
        return metrics_data

    def _calculate_head_metrics(self, attn_matrix):
        """Calculate entropy, sparsity, and average distance for an attention head"""
        # Row-normalize
        attn_matrix = attn_matrix / (attn_matrix.sum(dim=1, keepdim=True) + 1e-9)
        seq_len = attn_matrix.size(0)
        # Entropy (average per row)
        entropy = - (attn_matrix * (attn_matrix + 1e-9).log()).sum(dim=1).mean().item()
        # Sparsity: fraction of entries below threshold
        threshold = 1e-4
        sparsity = (attn_matrix < threshold).float().mean().item()
        # Weighted average distance from diagonal
        positions = torch.arange(seq_len).float()
        i_pos = positions.unsqueeze(1).expand(seq_len, seq_len)
        j_pos = positions.unsqueeze(0).expand(seq_len, seq_len)
        distances = torch.abs(i_pos - j_pos)
        total_weight = attn_matrix.sum().item()
        if total_weight > 0:
            avg_distance = (attn_matrix * distances).sum().item() / total_weight
        else:
            avg_distance = 0.0
        return {
            'entropy': entropy,
            'sparsity': sparsity,
            'distance': avg_distance
        }

def load_model(model_path, device):
    """Load the trained model"""
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    model = EnhancedCharTransformer(
        vocab_size=256,
        d_model=512,
        nhead=8,
        num_layers=12,
        dim_feedforward=2048,
        dropout=0.1
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    return model

def load_validation_sequences(data_path, num_seqs=10, seq_len=1024):
    """Load the first num_seqs validation sequences of length seq_len from enwik8 data."""
    # Assumes data_path points to the enwik8 file
    with open(data_path, 'rb') as f:
        data = f.read()
    # Use last 10% as validation
    split_idx = int(len(data) * 0.9)
    val_data = np.frombuffer(data[split_idx:], dtype=np.uint8)
    # Extract sequences
    sequences = []
    for i in range(num_seqs):
        start = i * seq_len
        end = start + seq_len
        if end > len(val_data):
            break
        seq = val_data[start:end]
        sequences.append(seq)
    if not sequences:
        raise ValueError("Not enough validation data for the requested number of sequences.")
    return torch.tensor(np.stack(sequences), dtype=torch.long)

def main():
    parser = argparse.ArgumentParser(description='Extract attention head metrics from real model attention')
    parser.add_argument('--model_path', type=str, default='../models/dense_char_transformer.pt',
                       help='Path to trained dense model')
    parser.add_argument('--data_path', type=str, default='../dense_transformer/data/enwik8',
                       help='Path to enwik8 data file')
    parser.add_argument('--seq_length', type=int, default=1024,
                       help='Sequence length for analysis')
    parser.add_argument('--num_seqs', type=int, default=10,
                       help='Number of validation sequences to use')
    parser.add_argument('--output', type=str, default='head_metrics.csv',
                       help='Output CSV file')
    args = parser.parse_args()
    print("Extracting real attention head metrics from model")
    print("=" * 50)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Load model
    model = load_model(args.model_path, device)
    # Load validation sequences
    print(f"Loading {args.num_seqs} validation sequences of length {args.seq_length} from {args.data_path}")
    val_seqs = load_validation_sequences(args.data_path, args.num_seqs, args.seq_length)
    # Extract metrics
    extractor = AttentionMetricsExtractor(model)
    print("Extracting attention metrics from real model attention...")
    metrics_data = extractor.extract_metrics(val_seqs)
    if not metrics_data:
        print("No attention metrics extracted!")
        return
    # Create DataFrame
    df = pd.DataFrame(metrics_data)
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"Saved {len(metrics_data)} head metrics to {args.output}")
    # Display summary
    print("\nSummary Statistics:")
    print("=" * 30)
    print(df.groupby('layer')[['entropy', 'sparsity', 'distance']].agg(['mean', 'std']).round(3))
    print(f"\nReady for Table 2.1 generation!")

if __name__ == "__main__":
    main() 