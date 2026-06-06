#!/usr/bin/env python
"""
POLYMER SMILES DIFFUSION MODEL - TRAINING SCRIPT
Implements masked diffusion language modeling for polymer SMILES generation.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import signal
import sys

warnings.filterwarnings('ignore')

# Import custom modules
from model import ConditionalDiffusionLM
from tokenizer import SMILESTokenizer


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_PATH = "/home/hice1/vyadav68/scratch/language_diffusion_model/PI1M_v2.csv"
OUTPUT_DIR = './pretrain_checkpoints'
PLOT_DIR = '/home/hice1/vyadav68/scratch/language_diffusion_model/plots'

# Model settings
MODEL_NAME = "answerdotai/ModernBERT-base"
USE_PROPERTY_CONDITIONING = True
MAX_LENGTH = 256

# Training hyperparameters
EPOCHS = 10
BATCH_SIZE = 256  # Increased from 128 - A100 can handle much more
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 512
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SAVE_STEPS = 1000
NUM_WORKERS = 8  # Reduced from 24 - too many workers can cause overhead

# Mixed precision training (CRITICAL for A100 performance)
USE_AMP = True  # Automatic Mixed Precision (FP16)
USE_COMPILE = True  # torch.compile for extra speed (PyTorch 2.0+)

# Resume from checkpoint (set to None to start fresh)
RESUME_FROM_STEP = None  # or specific step number like 9067


# ============================================================================
# DEVICE SETUP
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# ============================================================================
# DATASET CLASS
# ============================================================================

class PI1MDataset(Dataset):
    """Dataset for polymer SMILES (unlabeled pretraining)."""
    
    def __init__(self, csv_path, tokenizer, max_length=256):
        print(f"Loading dataset from: {csv_path}")
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Find SMILES column
        if 'p-SMILES' in self.data.columns:
            self.smiles = self.data['p-SMILES'].values
        elif 'SMILES' in self.data.columns:
            self.smiles = self.data['SMILES'].values
        else:
            self.smiles = self.data.iloc[:, 0].values
        
        print(f"Loaded {len(self.smiles)} SMILES structures")
        print(f"Sample SMILES: {self.smiles[0]}")
        
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smiles = str(self.smiles[idx])
        encoded = self.tokenizer.encode(smiles, add_special_tokens=True)
        
        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
        }


class DiffusionCollator:
    """Data collator for masked diffusion training with variable masking ratio."""
    
    def __init__(self, mask_token_id, pad_token_id, bos_token_id, eos_token_id):
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        batch_size, seq_len = input_ids.shape
        
        # Sample random masking ratio for each sample (0 to 1)
        mask_ratios = torch.rand(batch_size)
        
        # Create masked inputs and targets
        masked_input_ids = input_ids.clone()
        targets = input_ids.clone()
        
        for i in range(batch_size):
            # Find valid token positions (not PAD, BOS, EOS)
            valid_positions = (input_ids[i] != self.pad_token_id) & \
                             (input_ids[i] != self.bos_token_id) & \
                             (input_ids[i] != self.eos_token_id)
            valid_indices = valid_positions.nonzero(as_tuple=True)[0]
            
            if len(valid_indices) > 0:
                num_to_mask = int(len(valid_indices) * mask_ratios[i])
                if num_to_mask > 0:
                    mask_indices = valid_indices[torch.randperm(len(valid_indices))[:num_to_mask]]
                    masked_input_ids[i, mask_indices] = self.mask_token_id
        
        return {
            'input_ids': masked_input_ids,
            'attention_mask': attention_mask,
            'labels': targets,
            'mask_ratios': mask_ratios,
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def plot_dataset_statistics(dataset, output_dir):
    """Analyze and plot SMILES length distribution."""
    smiles_lengths = [len(str(dataset.smiles[i])) for i in range(min(1000, len(dataset)))]
    
    plt.figure(figsize=(10, 5))
    plt.hist(smiles_lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('SMILES Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of SMILES Lengths in Dataset', fontsize=14)
    plt.axvline(x=np.mean(smiles_lengths), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(smiles_lengths):.1f}')
    plt.axvline(x=np.median(smiles_lengths), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(smiles_lengths):.1f}')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/smiles_length_distribution.png")
    plt.close()
    
    print(f"\n--- Dataset Statistics ---")
    print(f"Average SMILES length: {np.mean(smiles_lengths):.1f} characters")
    print(f"Median SMILES length: {np.median(smiles_lengths):.1f} characters")
    print(f"Min/Max SMILES length: {min(smiles_lengths)} / {max(smiles_lengths)}")
    print(f"Std deviation: {np.std(smiles_lengths):.1f}")


def plot_training_progress(history, output_dir, epoch=None):
    """Plot training loss and learning rate curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    # Loss curve
    ax1.plot(history['steps'], history['train_loss'], alpha=0.4, 
             label='Raw Loss', linewidth=0.5, color='lightblue')
    
    window = min(100, len(history['train_loss']) // 10)
    if len(history['train_loss']) > window:
        smoothed = np.convolve(history['train_loss'], 
                               np.ones(window)/window, mode='valid')
        ax1.plot(history['steps'][window-1:], smoothed, 
                linewidth=2, label=f'Smoothed (window={window})', color='blue')
    
    ax1.set_xlabel('Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    title = f'Training Loss (Epoch {epoch})' if epoch else 'Training Loss'
    ax1.set_title(title, fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Learning rate
    ax2.plot(history['steps'], history['learning_rate'], color='orange', linewidth=2)
    ax2.set_xlabel('Steps', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/latest_training_progress.png")
    if epoch:
        plt.savefig(f"{output_dir}/snapshot_epoch_{epoch}.png", dpi=150)
    plt.close()


def save_checkpoint(model, optimizer, scheduler, history, global_step, epoch, 
                    output_dir, loss):
    """Save model checkpoint."""
    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'step': global_step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'vocab_size': len(tokenizer),
        'history': history,
    }, save_path / "model.pt")
    
    return save_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint and return training state."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    history = checkpoint.get('history', {
        'train_loss': [],
        'learning_rate': [],
        'steps': []
    })
    
    return checkpoint['step'], checkpoint['epoch'], history, checkpoint['loss']


# ============================================================================
# INTERRUPT HANDLER
# ============================================================================

def create_interrupt_handler(model, optimizer, scheduler, history, output_dir):
    """Create handler to save checkpoint on interrupt."""
    def save_on_interrupt(signum, frame):
        print("\n\n🛑 Interrupt detected! Saving checkpoint...")
        
        global_step = history['steps'][-1] if history['steps'] else 0
        epoch = len(set(history.get('epochs', [0])))
        loss = history['train_loss'][-1] if history['train_loss'] else 0
        
        interrupt_path = save_checkpoint(
            model, optimizer, scheduler, history, 
            global_step, epoch, output_dir, loss
        )
        interrupt_path = interrupt_path.parent / f"{interrupt_path.name}-interrupted"
        interrupt_path.mkdir(exist_ok=True)
        
        print(f"✓ Emergency checkpoint saved to: {interrupt_path}")
        sys.exit(0)
    
    return save_on_interrupt


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================


# Create output directories
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

# Load and prepare data
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

data = pd.read_csv(DATA_PATH)
smiles_list = data['SMILES'].values

# Initialize tokenizer
tokenizer = SMILESTokenizer(max_length=MAX_LENGTH)
tokenizer.build_vocab_from_data(smiles_list, min_freq=2)
vocab_size = len(tokenizer)

print(f"\nVocabulary size: {vocab_size}")
print(f"Max sequence length: {tokenizer.max_length}")

# Create dataset
dataset = PI1MDataset(DATA_PATH, tokenizer, MAX_LENGTH)
print(f"Dataset size: {len(dataset)}")

# Plot dataset statistics
plot_dataset_statistics(dataset, PLOT_DIR)

# Test tokenizer
test_smiles = "*CC(*)c1ccccc1"
encoded = tokenizer.encode(test_smiles)
print(f"\n--- Tokenizer Test ---")
print(f"Original SMILES: {test_smiles}")
print(f"Decoded SMILES: {tokenizer.decode(encoded['input_ids'])}")

# Create DataLoader
special_tokens = tokenizer.get_special_token_ids()
collator = DiffusionCollator(
    mask_token_id=special_tokens['mask'],
    pad_token_id=special_tokens['pad'],
    bos_token_id=special_tokens['bos'],
    eos_token_id=special_tokens['eos']
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=4,  # Increased from 2
    persistent_workers=True  # Keep workers alive between epochs
)

print(f"\nDataLoader: {len(dataloader)} batches per epoch")

# Initialize model
print("\n" + "="*60)
print("INITIALIZING MODEL")
print("="*60)

model = ConditionalDiffusionLM(
    model_name=MODEL_NAME,
    vocab_size=vocab_size,
    use_property_conditioning=USE_PROPERTY_CONDITIONING,
    dropout=0.1
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size (fp16): {total_params * 2 / 1e9:.2f} GB")

# Compile model for better performance (PyTorch 2.0+)
if USE_COMPILE:
    try:
        print("\nCompiling model with torch.compile()...")
        model = torch.compile(model, mode='reduce-overhead')
        
        # CRITICAL: Test compilation with a dummy batch
        print("Testing compiled model with dummy forward pass...")
        with torch.no_grad():
            dummy_input = torch.randint(0, vocab_size, (2, 256), device=device)
            dummy_mask = torch.ones(2, 256, device=device)
            dummy_timestep = torch.rand(2, device=device)
            _ = model(dummy_input, dummy_mask, dummy_timestep, None)
        
        print("✓ Model compiled and tested successfully!")
        
    except Exception as e:
        print(f"⚠ Could not compile model: {e}")
        print("Continuing without compilation...")
        USE_COMPILE = False
        # Reinitialize model without compilation
        model = ConditionalDiffusionLM(
            model_name=MODEL_NAME,
            vocab_size=vocab_size,
            use_property_conditioning=USE_PROPERTY_CONDITIONING,
            dropout=0.1
        ).to(device)

# Setup training
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY
)

total_steps = len(dataloader) * EPOCHS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=total_steps,
    eta_min=1e-7
)

criterion = nn.CrossEntropyLoss(ignore_index=special_tokens['pad'])

# Mixed precision training scaler
scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

history = {
    'train_loss': [],
    'learning_rate': [],
    'steps': []
}

global_step = 0
start_epoch = 0

# Load checkpoint if resuming
if RESUME_FROM_STEP is not None:
    checkpoint_path = Path(OUTPUT_DIR) / f"checkpoint-{RESUME_FROM_STEP}" / "model.pt"
    
    if checkpoint_path.exists():
        print("\n" + "="*60)
        print(f"RESUMING FROM CHECKPOINT: {RESUME_FROM_STEP}")
        print("="*60 + "\n")
        
        global_step, start_epoch, history, last_loss = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, device
        )
        
        print(f"✓ Resumed from step {global_step}, epoch {start_epoch + 1}")
        print(f"✓ Last loss: {last_loss:.4f}")
        print(f"✓ Current learning rate: {scheduler.get_last_lr()[0]:.2e}\n")
    else:
        print(f"⚠ Checkpoint not found, starting fresh\n")
        RESUME_FROM_STEP = None

# Register interrupt handler
interrupt_handler = create_interrupt_handler(
    model, optimizer, scheduler, history, OUTPUT_DIR
)
signal.signal(signal.SIGINT, interrupt_handler)
signal.signal(signal.SIGTERM, interrupt_handler)

# Training loop
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
print(f"Epochs: {EPOCHS} (starting from {start_epoch + 1})")
print(f"Batch size: {BATCH_SIZE}")
print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Mixed precision (AMP): {USE_AMP}")
print(f"Model compiled: {USE_COMPILE}")
print(f"Total steps: {total_steps:,}")
print(f"Current step: {global_step:,}")
print(f"Remaining steps: {total_steps - global_step:,}")
print(f"Save every: {SAVE_STEPS} steps\n")

model.train()

for epoch in range(start_epoch, EPOCHS):
    epoch_loss = 0.0
    epoch_start_step = global_step
    
    # Calculate batches to skip if resuming mid-epoch
    batches_per_epoch = len(dataloader)
    skip_batches = 0
    if RESUME_FROM_STEP and epoch == start_epoch:
        skip_batches = global_step % batches_per_epoch
    
    pbar = tqdm(
        enumerate(dataloader), 
        desc=f"Epoch {epoch+1}/{EPOCHS}",
        total=batches_per_epoch
    )
    
    for batch_idx, batch in pbar:
        if batch_idx < skip_batches:
            continue
        
        # Move batch to device
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        mask_ratios = batch['mask_ratios'].to(device, non_blocking=True)
        
        # Mixed precision training
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            # Forward pass
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                timestep=mask_ratios,
                property=None
            )
            
            # Compute loss
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
            loss = loss / GRADIENT_ACCUMULATION_STEPS  # Scale loss for accumulation
        
        # Backward pass with gradient scaling
        if USE_AMP:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights every GRADIENT_ACCUMULATION_STEPS
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            if USE_AMP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
            
            optimizer.zero_grad()
            scheduler.step()
            
            # Logging (only on actual weight updates)
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            global_step += 1
            current_lr = scheduler.get_last_lr()[0]
            
            pbar.set_description(
                f"Epoch {epoch+1}/{EPOCHS} | Global: {global_step}/{total_steps} "
                f"({100*global_step/total_steps:.1f}%)"
            )
            pbar.set_postfix({
                'loss': f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}", 
                'lr': f"{current_lr:.2e}"
            })
            
            history['train_loss'].append(loss.item() * GRADIENT_ACCUMULATION_STEPS)
            history['learning_rate'].append(current_lr)
            history['steps'].append(global_step)
            
            # Save checkpoint
            if global_step % SAVE_STEPS == 0:
                save_path = save_checkpoint(
                    model, optimizer, scheduler, history,
                    global_step, epoch, OUTPUT_DIR, loss.item() * GRADIENT_ACCUMULATION_STEPS
                )
                print(f"\n✓ Checkpoint saved to {save_path}")
    
    # Epoch summary
    num_weight_updates = (batches_per_epoch - skip_batches) // GRADIENT_ACCUMULATION_STEPS
    avg_loss = epoch_loss / num_weight_updates if num_weight_updates > 0 else 0
    steps_this_epoch = global_step - epoch_start_step
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{EPOCHS} Summary:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Steps this epoch: {steps_this_epoch}")
    print(f"  Total Steps: {global_step:,}/{total_steps:,}")
    print(f"  Progress: {100 * global_step / total_steps:.1f}%")
    print(f"{'='*60}\n")
    
    # Plot progress
    if (epoch + 1) % 2 == 0 or epoch == EPOCHS - 1:
        plot_training_progress(history, PLOT_DIR, epoch + 1)

# Save final model
print("\n" + "="*60)
print("SAVING FINAL MODEL")
print("="*60)

final_path = Path(OUTPUT_DIR) / "final_model"
final_path.mkdir(parents=True, exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'tokenizer_vocab': tokenizer.vocab,
    'training_history': history,
    'total_steps': global_step,
    'epochs': EPOCHS,
    'final_loss': history['train_loss'][-1],
}, final_path / "model.pt")

plot_training_progress(history, final_path)

print(f"✓ Model saved to: {final_path / 'model.pt'}")
print(f"✓ Plots saved to: {final_path / 'training_curves.png'}")
print(f"\nFinal loss: {history['train_loss'][-1]:.4f}")
print(f"Best loss: {min(history['train_loss']):.4f}")
print(f"Avg loss (last 100): {np.mean(history['train_loss'][-100:]):.4f}")
print("\n🎉 TRAINING COMPLETE! 🎉\n")