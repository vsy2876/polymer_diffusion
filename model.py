# model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import numpy as np

class GaussianFourierProjection(nn.Module):
    """Expands a single scalar into a rich frequency embedding."""
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Freeze weights
        self.W = nn.Parameter(torch.randn(1, embed_dim // 2) * scale, requires_grad=False)
        
    def forward(self, x):
        # x is (batch, 1)
        x_proj = x * self.W.to(x.device) * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ConditionalDiffusionLM(nn.Module):
    """Transformer-based discrete diffusion language model for conditional SMILES generation."""
    
    def __init__(self, model_name: str = "answer-ai/ModernBERT-base", 
                 vocab_size: int = None, 
                 use_property_conditioning: bool = False,
                 dropout: float = 0.1):
        """
        Initialize the Conditional Diffusion Language Model.
        
        Args:
            model_name: HuggingFace model identifier for the backbone
            vocab_size: Size of token vocabulary (if None, uses model's default)
            use_property_conditioning: Whether to use property conditioning (e.g., Tg, Egc)
            dropout: Dropout probability
        """
        super().__init__()
        
        # Load pretrained transformer backbone
        config = AutoConfig.from_pretrained(model_name)
        if vocab_size is not None:
            config.vocab_size = vocab_size
            # Set pad_token_id to 0 (assuming your tokenizer uses 0 for [PAD])
            config.pad_token_id = 0

        self.backbone = AutoModel.from_pretrained(
            model_name, 
            config=config, 
            ignore_mismatched_sizes=True
        )
        self.hidden_size = config.hidden_size
        
        # Property conditioning (for Egc, Tg, etc.)
        self.use_property_conditioning = use_property_conditioning
        if use_property_conditioning:
            self.property_embedding = nn.Sequential(
                GaussianFourierProjection(128),  # Transforms (batch, 1) -> (batch, 128)
                nn.Linear(128, self.hidden_size),
                nn.SiLU(), # SiLU is generally better than ReLU for diffusion
                nn.Linear(self.hidden_size, self.hidden_size)
            )
        
        # Timestep (noise level) embedding for diffusion process
        self.timestep_embedding = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, self.hidden_size)
        )
        
        # Output head for token prediction
        self.output_head = nn.Linear(self.hidden_size, vocab_size)
        
    def forward(self, input_ids, attention_mask, timestep, property=None):
        """
        Forward pass through the diffusion model.
        
        Args:
            input_ids: (batch, seq_len) masked token IDs
            attention_mask: (batch, seq_len) attention mask (1 for valid, 0 for padding)
            timestep: (batch, 1) current noise level in [0, 1]
            property: (batch, 1) optional property value for conditioning (normalized)
        
        Returns:
            logits: (batch, seq_len, vocab_size) prediction logits for all positions
        """
        batch_size, seq_len = input_ids.shape
        
        # Get transformer embeddings
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        
        # Add timestep conditioning (broadcast to all positions)
        t_embed = self.timestep_embedding(timestep.unsqueeze(-1))  # (batch, hidden_size)
        hidden_states = hidden_states + t_embed.unsqueeze(1)  # (batch, seq_len, hidden_size)
        
        # Add property conditioning if available
        if self.use_property_conditioning and property is not None:
            prop_embed = self.property_embedding(property.unsqueeze(-1))  # (batch, hidden_size)
            hidden_states = hidden_states + prop_embed.unsqueeze(1)
        
        # Predict tokens for all positions
        logits = self.output_head(hidden_states)  # (batch, seq_len, vocab_size)
        
        return logits