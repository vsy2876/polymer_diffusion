#!/usr/bin/env python
"""
POLYMER SMILES DIFFUSION MODEL - FINE-TUNING SCRIPT
Fine-tunes the pretrained model with property conditioning (Egc - band gap).
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import tempfile

# Force all Python temporary sandboxes to use PACE Phoenix Scratch space
if "SCRM" in os.environ:
    tempfile.tempdir = os.environ["SCRM"]
elif "SCRATCH" in os.environ:
    tempfile.tempdir = os.environ["SCRATCH"]
    
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
PRETRAIN_CHECKPOINT = "./pretrain_checkpoints/final_model/model.pt"  # or specific checkpoint
FINETUNING_DATA_PATH = "/home/hice1/vyadav68/scratch/language_diffusion_model/Egc.csv"
OUTPUT_DIR = './finetune_checkpoints'
PLOT_DIR = '/home/hice1/vyadav68/scratch/language_diffusion_model/plots_finetune'

# Model settings
MODEL_NAME = "answerdotai/ModernBERT-base"
USE_PROPERTY_CONDITIONING = True  # CRITICAL: Enable for fine-tuning
MAX_LENGTH = 256

# Fine-tuning hyperparameters (typically smaller LR than pretraining)
EPOCHS = 20  # More epochs for fine-tuning on smaller dataset
BATCH_SIZE = 64  # Smaller batch for smaller dataset
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 256
LEARNING_RATE = 1e-5  # Smaller LR for fine-tuning (was 5e-5 for pretraining)
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SAVE_STEPS = 100  # Save more frequently
NUM_WORKERS = 8

# Mixed precision training
USE_AMP = True
USE_COMPILE = True

# Resume from checkpoint
RESUME_FROM_STEP = None


# ============================================================================
# DEVICE SETUP
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# ============================================================================
# DATASET CLASS WITH PROPERTY
# ============================================================================

class PropertyDataset(Dataset):
    """Dataset for polymer SMILES with property conditioning."""
    
    def __init__(self, csv_path, tokenizer, max_length=256, property_col='Egc'):
        print(f"Loading property dataset from: {csv_path}")
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.property_col = property_col
        
        # Get SMILES and property
        self.smiles = self.data['SMILES'].values
        self.properties = self.data[property_col].values
        
        # Normalize properties to [0, 1] range for better training
        self.prop_min = self.properties.min()
        self.prop_max = self.properties.max()
        self.properties_normalized = (self.properties - self.prop_min) / (self.prop_max - self.prop_min)
        
        print(f"Loaded {len(self.smiles)} SMILES with properties")
        print(f"Property '{property_col}' range: [{self.prop_min:.4f}, {self.prop_max:.4f}]")
        print(f"Sample SMILES: {self.smiles[0]}")
        print(f"Sample property: {self.properties[0]:.4f} (normalized: {self.properties_normalized[0]:.4f})")
        
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smiles = str(self.smiles[idx])
        property_val = self.properties_normalized[idx]
        
        encoded = self.tokenizer.encode(smiles, add_special_tokens=True)
        
        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'property': torch.tensor(property_val, dtype=torch.float32),
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
        properties = torch.stack([item['property'] for item in batch])
        
        batch_size, seq_len = input_ids.shape
        
        # Sample random masking ratio for each sample (0 to 1)
        mask_ratios = torch.rand(batch_size)
        
        cfg_drop_mask = torch.rand(batch_size) < 0.15 # 15% dropout rate
        properties[cfg_drop_mask] = -1.0 # Set to unconditional dummy value
        
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
            'properties': properties,  # Add properties to batch
        }


# ============================================================================
# UTILITY FUNCTIONS (Same as pretraining)
# ============================================================================

def plot_property_distribution(properties, output_dir):
    """Plot distribution of property values."""
    plt.figure(figsize=(10, 5))
    plt.hist(properties, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Band Gap (Egc)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Band Gap Values in Fine-tuning Dataset', fontsize=14)
    plt.axvline(x=np.mean(properties), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(properties):.4f}')
    plt.axvline(x=np.median(properties), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(properties):.4f}')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/property_distribution.png")
    plt.close()
    
    print(f"\n--- Property Statistics ---")
    print(f"Mean: {np.mean(properties):.4f}")
    print(f"Median: {np.median(properties):.4f}")
    print(f"Min: {np.min(properties):.4f}")
    print(f"Max: {np.max(properties):.4f}")
    print(f"Std: {np.std(properties):.4f}")


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
    title = f'Fine-tuning Loss (Epoch {epoch})' if epoch else 'Fine-tuning Loss'
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
                    output_dir, loss, tokenizer):
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
        'tokenizer_vocab': tokenizer.vocab,
        'history': history,
    }, save_path / "model.pt")
    
    return save_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint and return training state."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    history = checkpoint.get('history', {
        'train_loss': [],
        'learning_rate': [],
        'steps': []
    })
    
    return checkpoint['step'], checkpoint.get('epoch', 0), history, checkpoint['loss']


def create_interrupt_handler(model, optimizer, scheduler, history, output_dir, tokenizer):
    """Create handler to save checkpoint on interrupt."""
    def save_on_interrupt(signum, frame):
        print("\n\n🛑 Interrupt detected! Saving checkpoint...")
        
        global_step = history['steps'][-1] if history['steps'] else 0
        epoch = len(set(history.get('epochs', [0])))
        loss = history['train_loss'][-1] if history['train_loss'] else 0
        
        interrupt_path = save_checkpoint(
            model, optimizer, scheduler, history, 
            global_step, epoch, output_dir, loss, tokenizer
        )
        
        print(f"✓ Emergency checkpoint saved to: {interrupt_path}")
        sys.exit(0)
    
    return save_on_interrupt


# ============================================================================
# MAIN FINE-TUNING
# ============================================================================

# Create output directories
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

# Load fine-tuning data
print("\n" + "="*60)
print("LOADING FINE-TUNING DATA")
print("="*60)

ft_data = pd.read_csv(FINETUNING_DATA_PATH)
ft_smiles_list = ft_data['SMILES'].values

# Load tokenizer vocab from pretrained checkpoint
print("\nLoading tokenizer vocab from pretrained checkpoint...")
pretrain_checkpoint = torch.load(PRETRAIN_CHECKPOINT, map_location=device)

if 'tokenizer_vocab' not in pretrain_checkpoint:
    raise ValueError("❌ 'tokenizer_vocab' not found in pretrained checkpoint! Cannot proceed.")

tokenizer = SMILESTokenizer(max_length=MAX_LENGTH)
tokenizer.vocab = pretrain_checkpoint['tokenizer_vocab']

# Rename * to [*] to handle Egc dataset notation
if '*' in tokenizer.vocab and '[*]' not in tokenizer.vocab:
    star_id = tokenizer.vocab['*']
    tokenizer.vocab['[*]'] = star_id
    del tokenizer.vocab['*']
    print(f"✅ Renamed '*' → '[*]' at ID {star_id}")

tokenizer.idx_to_token = {int(v): k for k, v in tokenizer.vocab.items()}
vocab_size = len(tokenizer)

print(f"✅ Loaded vocab from pretrained checkpoint ({vocab_size} tokens)")

print(f"Vocabulary size: {vocab_size}")
print(f"Max sequence length: {tokenizer.max_length}")

# Create property dataset
dataset = PropertyDataset(FINETUNING_DATA_PATH, tokenizer, MAX_LENGTH, property_col='Egc')
print(f"Fine-tuning dataset size: {len(dataset)}")

# Plot property distribution
plot_property_distribution(dataset.properties, PLOT_DIR)

# Test tokenizer
test_smiles = ft_smiles_list[0]
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
    prefetch_factor=4,
    persistent_workers=True
)

print(f"DataLoader: {len(dataloader)} batches per epoch")

# Initialize model
print("\n" + "="*60)
print("INITIALIZING MODEL WITH PROPERTY CONDITIONING")
print("="*60)

model = ConditionalDiffusionLM(
    model_name=MODEL_NAME,
    vocab_size=vocab_size,
    use_property_conditioning=USE_PROPERTY_CONDITIONING,  # MUST be True
    dropout=0.1
).to(device)

# Load pretrained weights
print(f"\nLoading pretrained weights from: {PRETRAIN_CHECKPOINT}")
if Path(PRETRAIN_CHECKPOINT).exists():
    # pretrain_checkpoint already loaded above for tokenizer vocab
    try:
        model.load_state_dict(pretrain_checkpoint['model_state_dict'], strict=False)
        print("✓ Loaded pretrained weights (non-strict to allow vocab/property differences)")
    except Exception as e:
        print(f"⚠ Could not load pretrained weights: {e}")
        print("Starting from scratch...")
else:
    print(f"⚠ Pretrained checkpoint not found at {PRETRAIN_CHECKPOINT}")
    print("Starting from random initialization...")

print("\n" + "="*60)
print("JOINT TRAINING: DIFFERENTIAL LEARNING RATES")
print("="*60)

# Make sure ALL parameters are unfrozen
for param in model.parameters():
    param.requires_grad = True

# Separate parameters into two groups
backbone_params = []
property_params = []

for name, param in model.named_parameters():
    if 'property' in name.lower() or 'cond' in name.lower():
        property_params.append(param)
    else:
        backbone_params.append(param)

print(f"Backbone parameters: {len(backbone_params)} tensors")
print(f"Property parameters: {len(property_params)} tensors")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {total_params:,}")
print(f"Model size (fp16): {total_params * 2 / 1e9:.2f} GB")

# Compile model
if USE_COMPILE:
    try:
        print("\nCompiling model with torch.compile()...")
        model = torch.compile(model, mode='default')
        
        # Test compilation
        print("Testing compiled model...")
        with torch.no_grad():
            dummy_input = torch.randint(0, vocab_size, (2, 256), device=device)
            dummy_mask = torch.ones(2, 256, device=device)
            dummy_timestep = torch.rand(2, device=device)
            dummy_prop = torch.rand(2, device=device)  # Add property
            _ = model(dummy_input, dummy_mask, dummy_timestep, dummy_prop)
        
        print("✓ Model compiled successfully!")
        
    except Exception as e:
        print(f"⚠ Compilation failed: {e}")
        print("Continuing without compilation...")
        USE_COMPILE = False

# Setup training with differential learning rates
# Overriding the global LEARNING_RATE here explicitly
optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 2e-6},    # Micro-LR to preserve SMILES grammar
    {'params': property_params, 'lr': 5e-5}     # Normal LR for property conditioning
], weight_decay=WEIGHT_DECAY)

print("\n✅ Optimizer configured:")
print("   - Backbone LR: 2e-6")
print("   - Property Head LR: 5e-5")


total_steps = len(dataloader) * EPOCHS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=total_steps,
    eta_min=1e-8  # Even smaller min LR for fine-tuning
)

criterion = nn.CrossEntropyLoss(ignore_index=special_tokens['pad'])
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

# Register interrupt handler
interrupt_handler = create_interrupt_handler(
    model, optimizer, scheduler, history, OUTPUT_DIR, tokenizer
)
signal.signal(signal.SIGINT, interrupt_handler)
signal.signal(signal.SIGTERM, interrupt_handler)

# Training loop
print("\n" + "="*60)
print("STARTING FINE-TUNING")
print("="*60)
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Mixed precision (AMP): {USE_AMP}")
print(f"Model compiled: {USE_COMPILE}")
print(f"Total steps: {total_steps:,}")
print(f"Save every: {SAVE_STEPS} steps\n")

model.train()

for epoch in range(start_epoch, EPOCHS):
    epoch_loss = 0.0
    epoch_start_step = global_step
    
    batches_per_epoch = len(dataloader)
    
    pbar = tqdm(
        enumerate(dataloader), 
        desc=f"Epoch {epoch+1}/{EPOCHS}",
        total=batches_per_epoch
    )
    
    for batch_idx, batch in pbar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        mask_ratios = batch['mask_ratios'].to(device, non_blocking=True)
        properties = batch['properties'].to(device, non_blocking=True)  # Property conditioning!
        
        # Mixed precision training
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            # Forward pass WITH property conditioning
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                timestep=mask_ratios,
                property=properties  # CRITICAL: Pass property values
            )
            
            # Compute loss
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
            loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        # Backward pass
        if USE_AMP:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights
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
            
            # Logging
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
                    global_step, epoch, OUTPUT_DIR, 
                    loss.item() * GRADIENT_ACCUMULATION_STEPS, tokenizer
                )
                print(f"\n✓ Checkpoint saved to {save_path}")
    
    # Epoch summary
    num_weight_updates = batches_per_epoch // GRADIENT_ACCUMULATION_STEPS
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
print("SAVING FINAL FINE-TUNED MODEL")
print("="*60)

final_path = Path(OUTPUT_DIR) / "final_model"
final_path.mkdir(parents=True, exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'tokenizer_vocab': tokenizer.vocab,
    'property_min': dataset.prop_min,  # Save for denormalization
    'property_max': dataset.prop_max,
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
print(f"\nProperty range for generation: [{dataset.prop_min:.4f}, {dataset.prop_max:.4f}]")
print("\n🎉 FINE-TUNING COMPLETE! 🎉\n")