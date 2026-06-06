# tokenizer.py

# Uses regex patterns to tokenize SMILES strings and build a vocabulary from the dataset.
# 


import re
from collections import Counter

class SMILESTokenizer:
    """Hybrid SMILES tokenizer with regex patterns and polymer support."""
    
    def __init__(self, max_length=256):
        self.max_length = max_length
        
        # Regex pattern for SMILES tokenization
        self.pattern = re.compile(r"""
            \[[^\]]+\]|           # Bracketed atoms: [*], [nH], [C@H], [NH3+]
            Br?|Cl?|Si|           # 2-char elements + Silicon (common in polymers)
            [BCNOPSFIbcnops]|     # Single atoms
            @@?|                  # Chirality
            [=\#]|                # Bonds
            %\d{2}|               # Ring numbers 10-99
            \d|                   # Ring numbers 0-9
            \*|                   # POLYMER NOTATION (crucial!)
            [()\/\\+\-]           # Branches, stereo, charges
        """, re.VERBOSE)
        
        # Initialize with special tokens
        self.vocab = {
            '[PAD]': 0,
            '[BOS]': 1, 
            '[EOS]': 2,
            '[MASK]': 3,
            '[UNK]': 4,  # For truly unknown tokens
        }
        self.special_token_count = len(self.vocab)
        
    def build_vocab_from_data(self, smiles_list, min_freq=2):
        """
        Build vocabulary from dataset (like the regex approach).
        
        Args:
            smiles_list: List of SMILES strings
            min_freq: Minimum frequency to include token
        """
        print(f"Building vocabulary from {len(smiles_list)} SMILES...")
        
        all_tokens = []
        for smi in smiles_list:
            tokens = self.pattern.findall(smi)
            all_tokens.extend(tokens)
        
        # Count frequencies
        token_counts = Counter(all_tokens)
        
        # Add tokens above frequency threshold
        idx = self.special_token_count
        for token, count in token_counts.most_common():
            if count >= min_freq:
                if token not in self.vocab:
                    self.vocab[token] = idx
                    idx += 1
        
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Top 20 tokens: {list(token_counts.most_common(20))}")
        
        return self
    
    def encode(self, smiles, add_special_tokens=True):
        """Tokenize SMILES using regex patterns."""
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.vocab['[BOS]'])
        
        # Extract tokens using regex
        smiles_tokens = self.pattern.findall(smiles)
        
        for token in smiles_tokens:
            # Use vocab if available, otherwise [UNK]
            token_id = self.vocab.get(token, self.vocab['[UNK]'])
            tokens.append(token_id)
        
        if add_special_tokens:
            tokens.append(self.vocab['[EOS]'])
        
        # Pad/truncate
        attention_mask = [1] * len(tokens)
        if len(tokens) < self.max_length:
            padding = self.max_length - len(tokens)
            tokens.extend([self.vocab['[PAD]']] * padding)
            attention_mask.extend([0] * padding)
        else:
            tokens = tokens[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        
        return {
            'input_ids': tokens,
            'attention_mask': attention_mask
        }
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Convert token IDs back to SMILES."""
        chars = []
        special = ['[PAD]', '[BOS]', '[EOS]', '[MASK]', '[UNK]']
        
        for tid in token_ids:
            token = self.idx_to_token.get(tid, '')
            if skip_special_tokens and token in special:
                continue
            chars.append(token)
        
        return ''.join(chars)
    
    def __len__(self):
        return len(self.vocab)
    
    def get_special_token_ids(self):
        return {
            'pad': self.vocab['[PAD]'],
            'bos': self.vocab['[BOS]'],
            'eos': self.vocab['[EOS]'],
            'mask': self.vocab['[MASK]'],
            'unk': self.vocab['[UNK]']
        }
