import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import math
import time
import wandb
import argparse
import numpy as np
import random
from stable_char_transformer import (
    EnhancedCharTransformer, 
    ByteTokenizer, 
    create_batches, 
    load_data,
    train_model
)

def visualize_loss(train_losses, val_losses=None, output_file='dense_model_loss.png'):
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
    perplexity_ticks = [math.exp(x) for x in loss_ticks if x > 0]
    ax2.set_yticks(perplexity_ticks)
    ax2.set_yticklabels([f'{x:.1f}' for x in perplexity_ticks])
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
    """Generate text using the trained model"""
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            prompt=input_tensor,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            tokenizer=tokenizer,
            device=device
        )
    
    # Decode and return the generated text
    return tokenizer.decode(output[0].tolist())

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Dense Character Transformer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq_length', type=int, default=1024, help='Sequence length')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Model configuration
    config = {
        'vocab_size': 256,  # Keep at 256 for byte-level tokenization
        'd_model': 512,     # Keep at 512 for our analysis
        'nhead': 8,
        'num_layers': 12,   # Keep at 12 for our analysis
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'activation_dropout': 0.1,
        'token_dropout': 0.05,
        'use_checkpoint': True,
        'stochastic_depth_prob': 0.1,
        'seed': args.seed,
        'num_epochs': args.num_epochs
    }
    
    # Initialize wandb
    run_name = args.wandb_run_name or f"dense_seed_{args.seed}"
    wandb.init(
        project="dense-transformer-training",
        config=config,
        name=run_name,
        dir='wandb_logs' # Set logging directory
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using seed: {args.seed}")
    
    # Create model instance
    model = EnhancedCharTransformer(**config)
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
    model_path = f'bachelor_thesis/models/dense_seed_{args.seed}.pt'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Saving model to {model_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config,
        'seed': args.seed
    }, model_path)
    
    # Visualize training history
    print("Generating loss plot...")
    visualize_loss(train_losses, val_losses, f'dense_seed_{args.seed}_training_loss.png')
    
    wandb.finish()

if __name__ == "__main__":
    main()