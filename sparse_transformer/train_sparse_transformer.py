import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import math
import argparse
import wandb
import time
import numpy as np
import random
from stable_char_transformer import (
    SparseTransformer, 
    ByteTokenizer, 
    create_batches, 
    load_data,
    train_model
)
from sparse_byte_transformer import SparseByteTransformer
from torch.cuda.amp import autocast

def visualize_loss(train_losses, val_losses=None, output_file='sparse_model_loss.png'):
    """Visualize training and validation losses"""
    plt.figure(figsize=(12, 6))
    
    # Plot training loss
    plt.plot(train_losses, label='Training Loss', marker='o', markersize=4, linestyle='-', linewidth=1)
    
    # Plot validation loss if available
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', marker='s', markersize=4, linestyle='-', linewidth=1)
        
        # Plot best validation loss point
        best_epoch = val_losses.index(min(val_losses))
        best_loss = val_losses[best_epoch]
        plt.plot(best_epoch, best_loss, 'r*', markersize=10, label=f'Best Val Loss: {best_loss:.4f}')
    
    plt.title('Model Training History', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add perplexity as secondary y-axis
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Create perplexity ticks based on loss values
    loss_ticks = ax1.get_yticks()
    perplexity_ticks = []
    for x_tick in loss_ticks:
        if x_tick > 0:
            try:
                perplexity_ticks.append(math.exp(x_tick))
            except OverflowError:
                perplexity_ticks.append(float('inf'))
        # else: # if x_tick <= 0, decide how to handle or skip
            # pass # or append a specific value if needed for plotting

    ax2.set_yticks(perplexity_ticks)
    ax2.set_yticklabels([f'{px:.1f}' if px != float('inf') else '>1e300' for px in perplexity_ticks])
    ax2.set_ylabel('Perplexity', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print("\nTraining Statistics:")
    print(f"Initial Loss: {train_losses[0]:.4f}")
    print(f"Final Loss: {train_losses[-1]:.4f}")
    print(f"Best Loss: {min(train_losses):.4f}")
    
    if val_losses:
        print("\nValidation Statistics:")
        print(f"Initial Loss: {val_losses[0]:.4f}")
        print(f"Final Loss: {val_losses[-1]:.4f}")
        print(f"Best Loss: {min(val_losses):.4f}")

def generate_text(model, tokenizer, prompt, max_length=1000, temperature=0.7, top_k=50, top_p=0.9, device='cuda'):
    """Generate text using the trained sparse transformer with improved sampling"""
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated = input_tensor
    
    # Track recent tokens for repetition detection
    recent_tokens = []
    token_counts = {}
    
    # Generate text token by token
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            with autocast():
                logits = model(generated)
                next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Dynamic repetition penalty based on recent usage
            for token in set(recent_tokens):
                count = recent_tokens.count(token)
                penalty = 1.0 + (count * 0.5)  # Increased penalty for frequency
                next_token_logits[token] /= penalty
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            
            # Update tracking
            recent_tokens.append(token_id)
            if len(recent_tokens) > 20:  # Track last 20 tokens
                recent_tokens.pop(0)
            
            token_counts[token_id] = token_counts.get(token_id, 0) + 1
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop conditions
            if token_id in [10, 0]:  # newline or end token
                break
                
            # Check for repetitive patterns
            if len(recent_tokens) >= 5:
                # Check for immediate repetition
                if len(set(recent_tokens[-5:])) == 1:
                    break
                    
                # Check for bi-gram repetition
                if len(recent_tokens) >= 10:
                    last_bigrams = [tuple(recent_tokens[i:i+2]) for i in range(len(recent_tokens)-2)]
                    if len(set(last_bigrams)) <= 2:
                        break
            
            # Check for overuse of any token
            max_count = max(token_counts.values()) if token_counts else 0
            if max_count > len(generated[0]) * 0.3:  # No token should be >30% of generation
                break
    
    # Decode and return the generated text
    generated_ids = generated[0].tolist()
    return tokenizer.decode(generated_ids)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Sparse Character Transformer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq_length', type=int, default=1024, help='Sequence length')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--wandb_project', type=str, default='sparse-transformer-training', help='WandB project name')
    parser.add_argument('--mask_subset', type=str, default='0123', help='String listing which cluster IDs to activate (e.g., "0", "03", "123")')
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

    # Initialize loss lists
    train_losses = []
    val_losses = []

    # Model configuration
    config_dict = {
        'vocab_size': 256,  # Keep at 256 for byte-level tokenization
        'd_model': 512,     # Keep at 512 for our analysis
        'nhead': 8,
        'num_layers': 12,   # Keep at 12 for our analysis
        'dim_feedforward': 2048,
        'dropout': 0.05,
        'attention_dropout': 0.1,
        'activation_dropout': 0.05,
        'token_dropout': 0.02,
        'use_checkpoint': True,
        'stochastic_depth_prob': 0.1,
        'seq_length': 1024,  # Added: SparseByteTransformer needs this for PositionalEncoding
        'seed': args.seed,
        'num_epochs': args.num_epochs,
        'mask_subset': args.mask_subset  # Added: For ablation study
    }
    
    # Initialize wandb
    run_name = args.wandb_run_name or f"sparse_seed_{args.seed}"
    wandb.init(
        project=args.wandb_project,
        config=config_dict,
        name=run_name,
        dir='wandb_logs' # Set logging directory
    )
    
    # Convert dict to Namespace for attribute access in the model
    model_config = argparse.Namespace(**config_dict)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using seed: {args.seed}")
    
    # Create model instance
    model = SparseByteTransformer(model_config)
    model = model.to(device)
    
    # Create tokenizer
    tokenizer = ByteTokenizer()
    
    # Load training data (always train for the statistical analysis)
    print("Loading training data...")
    train_text = load_data('data/enwik8')
    
    # Split into train/val
    split_idx = int(len(train_text) * 0.9)
    train_data = tokenizer.encode(train_text[:split_idx])
    val_data = tokenizer.encode(train_text[split_idx:])
    
    # Create batches
    train_batches = create_batches(train_data, args.batch_size, args.seq_length)
    val_batches = create_batches(val_data, args.batch_size, args.seq_length)
    
    # Train model
    print("Training model...")
    start_training_time = time.time()
    
    model, (train_losses, val_losses) = train_model(
        model=model,
        train_batches=train_batches,
        val_batches=val_batches,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.1,
        warmup_steps=1000,
        device=device,
        patience=8,  # Increased patience for longer training
        min_lr=1e-5,  # Minimum learning rate
        gradient_accumulation_steps=4,  # Gradient accumulation for stability
        use_mixed_precision=True,  # Use mixed precision training
        use_cosine_schedule=True,  # Use cosine learning rate schedule
        use_wandb=True # Enable wandb logging
    )
    
    end_training_time = time.time()
    total_training_time = end_training_time - start_training_time
    
    # Calculate final metrics
    final_val_loss = val_losses[-1] if val_losses else train_losses[-1]
    final_val_ppl = math.exp(final_val_loss) if final_val_loss < 700 else float('inf')
    
    # Calculate tokens per second (approximate)
    total_tokens = len(train_data) * args.num_epochs
    tokens_per_sec = total_tokens / total_training_time
    
    # Get peak GPU memory
    peak_gpu_mem_MB = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    
    # Log final metrics to wandb
    wandb.run.summary["final_val_loss"] = final_val_loss
    wandb.run.summary["final_val_ppl"] = final_val_ppl
    wandb.run.summary["tokens_per_sec"] = tokens_per_sec
    wandb.run.summary["peak_gpu_mem_MB"] = peak_gpu_mem_MB
    
    print(f"Training completed!")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final validation perplexity: {final_val_ppl:.2f}")
    print(f"Tokens per second: {tokens_per_sec:.0f}")
    print(f"Peak GPU memory: {peak_gpu_mem_MB:.1f} MB")
    
    # Save model
    model_path = f'bachelor_thesis/models/sparse_seed_{args.seed}.pt'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Saving model to {model_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config_dict,
        'seed': args.seed
    }, model_path)
    
    # Visualize training history
    print("Generating loss plot...")
    visualize_loss(train_losses, val_losses, f'sparse_seed_{args.seed}_training_loss.png')
    
    wandb.finish()

if __name__ == "__main__":
    main()