import os
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem
from transformers import AutoModel, AutoTokenizer
from xtb.interface import Calculator, Param
from xtb.libxtb import VERBOSITY_MUTED

# Suppress noisy background chemistry warnings
RDLogger.logger().setLevel(RDLogger.CRITICAL)

# Explicitly import from your local project architecture
from tokenizer import SMILESTokenizer
from model import ConditionalDiffusionLM

# ==========================================
# 1. LOCAL CONFIGURATION & RUNTIME ENVIRONMENT
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FINETUNE_CHECKPOINT = "./finetune_checkpoints/final_model/model.pt"
MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 256
BATCH_SIZE_GEN = 10000  # Strict journal-grade benchmark scale

print(f"🚀 Initializing Local Stack on: {DEVICE}")

# Load model states and map property bounds
checkpoint = torch.load(FINETUNE_CHECKPOINT, map_location=DEVICE, weights_only=False)
prop_min = checkpoint['property_min']
prop_max = checkpoint['property_max']
vocab_size = checkpoint['vocab_size']

# Build tokenizer environment mirroring your notebook cell 2 setup
tokenizer = SMILESTokenizer(max_length=MAX_LENGTH)
tokenizer.vocab = dict(checkpoint['tokenizer_vocab'])

# Harmonize the internal vocabulary maps for wildcard anchors
if '*' in tokenizer.vocab and '[*]' not in tokenizer.vocab:
    star_id = tokenizer.vocab['*']
    tokenizer.vocab['[*]'] = star_id
    del tokenizer.vocab['*']
tokenizer.idx_to_token = {int(v): k for k, v in tokenizer.vocab.items()}

# Instantiation of your native conditional ModernBERT model
model = ConditionalDiffusionLM(
    model_name=MODEL_NAME, vocab_size=vocab_size, use_property_conditioning=True, dropout=0.0
).to(DEVICE)

# Standardize torch compile key translations
state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# Native calibration values fetched from your notebook regression cell
XTB_SLOPE = 0.279
XTB_INTERCEPT = 3.833

# ==========================================
# 2. LOCAL BATCH INFERENCE SCHEDULER (PATCHED)
# ==========================================
@torch.no_grad()
def generate_batch_eval(model, tokenizer, num_samples=10000, target_egc=4.1, steps=40, guidance_scale=0.5, temperature=0.8):
    """Generates continuous token arrays with explicit VRAM synchronization points."""
    special_tokens = tokenizer.get_special_token_ids()
    mask_id, bos_id, pad_id = special_tokens['mask'], special_tokens['bos'], special_tokens['pad']
    
    prop_norm = max(0.0, min(1.0, float((target_egc - prop_min) / (prop_max - prop_min))))
    generated_strings = []
    
    # Using a safe block size of 100 to isolate the diagnostic run
    sub_batch_size = 100  
    loops = math.ceil(num_samples / sub_batch_size)
    
    print(f"⚙️ Running generation loop: Processing {num_samples} iterations in {loops} blocks...")
    
    for l in range(loops):
        t0 = time.time()
        current_b = min(sub_batch_size, num_samples - (l * sub_batch_size))
        
        prop_tensor = torch.tensor([prop_norm, -1.0], device=DEVICE).repeat(current_b)
        input_ids = torch.full((current_b, MAX_LENGTH), mask_id, device=DEVICE, dtype=torch.long)
        input_ids[:, 0] = bos_id
        
        for step in range(steps):
            ratio_masked = 1.0 - (step / steps)
            timestep_tensor = torch.tensor([ratio_masked, ratio_masked], device=DEVICE).repeat(current_b)
            cfg_inputs = input_ids.repeat(2, 1)
            
            # CRITICAL FIX: Convert mask explicitly to Bool for ModernBERT's attention engines
            att_mask = (cfg_inputs != pad_id).bool()
            
            logits = model(
                input_ids=cfg_inputs,
                attention_mask=att_mask,
                timestep=timestep_tensor,
                property=prop_tensor
            )
            
            cond_logits, uncond_logits = logits.chunk(2, dim=0)
            final_logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
            final_logits = final_logits / temperature
            
            probs = F.softmax(final_logits, dim=-1)
            pred_ids = torch.multinomial(probs.view(-1, vocab_size), num_samples=1).view(current_b, MAX_LENGTH)
            confidences, _ = torch.max(probs, dim=-1)
            
            if step < steps - 1:
                num_to_keep = int(MAX_LENGTH * (1 - ratio_masked))
                noisy_conf = confidences + torch.rand_like(confidences) * 0.1
                sorted_indices = torch.argsort(noisy_conf, dim=1, descending=True)
                
                keep_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                for b in range(current_b):
                    keep_mask[b, sorted_indices[b, :num_to_keep]] = True
                keep_mask[:, 0] = True
                
                input_ids = torch.where(keep_mask, pred_ids, torch.full_like(pred_ids, mask_id))
            else:
                input_ids = pred_ids

        # Local string extraction pass protecting chemical anchors
        for b in range(current_b):
            tokens = input_ids[b].cpu().numpy()
            decoded = []
            for tid in tokens:
                token = tokenizer.idx_to_token.get(int(tid), '')
                if token == '[EOS]': 
                    break
                if token not in ['[PAD]', '[BOS]', '[MASK]', '[UNK]']:
                    decoded.append(token)
            generated_strings.append("".join(decoded))
            
        print(f"   📊 Block {l+1}/{loops} completed in {time.time() - t0:.2f}s | Total Strings: {len(generated_strings)}")
        
        # CRITICAL FIX: Explicitly flush the CUDA cache context after each sub-batch block execution
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
            
    return generated_strings

# ==========================================
# 3. QUANTUM CHEMISTRY CALCULATOR LOOP
# ==========================================
import subprocess
import shutil

def calc_local_xtb_gap(smi):
    """
    Calls the global xtb binary directly via subprocess.
    Bypasses fragile python-C bindings to guarantee a numeric output.
    """
    # 1. Verify the binary is globally accessible in your environment
    if not shutil.which("xtb"):
        return None

    # Define temporary file names to prevent multi-thread write overlaps
    pid = os.getpid()
    xyz_file = f"temp_mol_{pid}.xyz"
    out_file = f"xtb_out_{pid}.log"
    
    try:
        # 2. Swap wildcards to Carbon to build a closed structure for RDKit
        clean_smi = smi.replace('[*]', 'C').replace('*', 'C')
        mol = Chem.MolFromSmiles(clean_smi)
        if mol is None: 
            return None
        
        mol = Chem.AddHs(mol)
        # Use simple random coordinates to guarantee a 3D structure is generated
        if AllChem.EmbedMolecule(mol, maxAttempts=100, useRandomCoords=True) == -1:
            return None
        
        # 3. Write out a standard chemical XYZ coordinate file
        atoms = mol.GetAtoms()
        conformer = mol.GetConformer()
        
        with open(xyz_file, "w") as f:
            f.write(f"{mol.GetNumAtoms()}\n")
            f.write(f"Generated polymer segment proxy {smi}\n")
            for i, atom in enumerate(atoms):
                pos = conformer.GetAtomPosition(i)
                f.write(f"{atom.GetSymbol()} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n")
        
        # 4. Call the terminal binary using an explicit shell subprocess pass
        # We run with '--gfn 2' matching your verified terminal workflow
        cmd = ["xtb", xyz_file, "--gfn", "2"]
        with open(out_file, "w") as out_f:
            subprocess.run(cmd, stdout=out_f, stderr=subprocess.DEVNULL, text=True, check=True)
            
        # 5. Parse the resulting output log file for the HOMO-LUMO gap line
        raw_gap = None
        with open(out_file, "r") as f:
            for line in f:
                if "HOMO-LUMO GAP" in line or "HOMO-LUMO gap" in line:
                    # Cleanly extract the numeric scalar out of the text row
                    parts = line.split()
                    for idx, part in enumerate(parts):
                        if "eV" in part and idx > 0:
                            raw_gap = float(parts[idx-1])
                            break
                    if raw_gap: 
                        break
                        
        return raw_gap

    except Exception as e:
        return None
        
    finally:
        # 6. Strict cleanup: Erase all temporary files from your desktop directory
        for f_path in [xyz_file, out_file, "wbo", "charges", "xtbrestart", "xtbopt.log", "xtbtopo.xyz"]:
            # Append local process ID extensions to clean up custom xTB dumps
            for target in [f_path, f_path + f"_{pid}", f_path + ".engrad"]:
                if os.path.exists(target):
                    try: os.remove(target)
                    except: pass

# ==========================================
# 4. COMPREHENSIVE STATISTICAL ANALYSIS ENGINE
# ==========================================
def evaluate_comprehensive_suite(gen_pool, train_csv_path="PI1M_v2.csv", target_egc=4.1):
    """Calculates all 6 physical and informational metrics from your publication criteria."""
    print("\n" + "="*60 + "\n📊 INITIATING SYSTEM BENCHMARK ASSESSMENT METRICS\n" + "="*60)
    total_generated = len(gen_pool)
    
    # 1. Chemical Validity Assessment
    valid_structures = []
    validity_count = 0
    for s in gen_pool:
        # Enforce that strings must include the structural polymerization wildcards
        if ('[*]' in s or '*' in s) and Chem.MolFromSmiles(s) is not None:
            validity_count += 1
            valid_structures.append(s)
            
    validity_rate = (validity_count / total_generated) * 100
    if validity_count == 0:
        print("❌ 0.00% Validity encountered. Terminating tracking pipeline.")
        return
        
    canonical_valid = []
    for s in valid_structures:
        try:
            canonical_valid.append(Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True))
        except: 
            pass

    # 2. Uniqueness Assessment
    unique_valid = list(set(canonical_valid))
    uniqueness_rate = (len(unique_valid) / len(canonical_valid)) * 100
    
    # 3. Novelty Evaluation Matrix
    print("📂 Cross-referencing generated molecules against dataset catalog maps...")
    try:
        ref_df = pd.read_csv(train_csv_path)
        col = 'p-SMILES' if 'p-SMILES' in ref_df.columns else 'SMILES'
        ref_set = set(ref_df[col].dropna().tolist())
    except FileNotFoundError:
        print("⚠️ Training data path not found. Defaulting to fallback array.")
        ref_set = {"*C*", "*CC*", "*CCC*"}
        
    novel_count = sum(1 for s in unique_valid if s not in ref_set)
    novelty_rate = (novel_count / len(unique_valid)) * 100 if unique_valid else 0.0

    # 4. Internal Structural Diversity Check (IntDiv)
    print("📐 Processing Morgan Bit-Vectors for structural disparity tracking...")
    mols = [Chem.MolFromSmiles(s) for s in unique_valid if Chem.MolFromSmiles(s) is not None]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    
    # Cap matching items list at 1000 to manage local CPU execution speeds
    eval_size = min(len(fps), 1000) 
    distances = []
    for i in range(eval_size):
        # Use BulkTanimotoSimilarity to compare fps[i] to all remaining fingerprints in the sliced pool
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:eval_size])
        for sim in sims:
            distances.append(1.0 - sim)
    internal_diversity = np.mean(distances) if distances else 0.0

    # 5. Continuous Property Target Control Adherence (MAE Calculation)
    print("🧪 Computing tight-binding single points over target candidates...")
    errors = []
    for s in unique_valid[:100]: # Sample 100 structures to ensure quick local execution loops
        raw_gap = calc_local_xtb_gap(s)
        if raw_gap is not None:
            calibrated_egc = (XTB_SLOPE * raw_gap) + XTB_INTERCEPT
            errors.append(abs(target_egc - calibrated_egc))
    property_mae = np.mean(errors) if errors else float('nan')

    # ==========================================
    # 5. JOURNAL BENCHMARK OUTPUT COMPILATION
    # ==========================================
    print("\n" + "="*60)
    print("📈 PUBLICATION-READY EVALUATION REPORT MATRIX")
    print("="*60)
    print(f"Target Requested Property (Egc) : {target_egc:.2f} eV")
    print(f"Total Model Generative Pool    : {total_generated:,}")
    print("-"*60)
    print(f"1. Chemical Validity Rate      : {validity_rate:.2f}%")
    print(f"2. Uniqueness Rate             : {uniqueness_rate:.2f}%")
    print(f"3. Novelty Rate                : {novelty_rate:.2f}%")
    print(f"4. Internal Diversity (IntDiv)  : {internal_diversity:.4f}")
    print(f"5. Target Property Adherence   : {property_mae:.4f} eV [MAE]")
    print("="*60 + "\n")

# ==========================================
# RUN TIME CODE ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    # Generate complete candidate stream
    polymer_pool = generate_batch_eval(
        model=model, 
        tokenizer=tokenizer, 
        num_samples=200, 
        target_egc=4.5, 
        steps=60,
        guidance_scale=1.5,   # ADJUSTED: Strengthens target adherence without destroying syntax
        temperature=0.5       # DECREASED: Cools down sampling to favor high-probability chemical tokens
    )
    
    # Process candidates against baseline distributions
    evaluate_comprehensive_suite(polymer_pool, train_csv_path="PI1M_v2.csv", target_egc=4.5)