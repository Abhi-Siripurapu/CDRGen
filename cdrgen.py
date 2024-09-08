import sys
import torch
import numpy as np
import os
import subprocess
from transformers import AutoModelForMaskedLM, AutoTokenizer
from abnumber import Chain
import torch.nn.functional as F
from antipasti.utils.torch_utils import load_checkpoint
from alphafold import generate_structure

def load_model(model_dir):
    """Load the pre-trained model and tokenizer."""
    model = AutoModelForMaskedLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def mask_cdr3(sequence):
    """Apply masking to the CDR3 region of the antibody sequence."""
    chain = Chain(sequence, scheme='chothia')
    cdr3 = chain.get_cdr('CDR3')
    if cdr3:
        start, end = cdr3.start_pos - 1, cdr3.end_pos  # Adjust for zero-based indexing
        masked_sequence = sequence[:start] + '[MASK]' * (end - start) + sequence[end:]
        return masked_sequence, (start, end)
    return sequence, None

def predict_masked_sequence(model, tokenizer, sequence, top_k=10):
    """Generate predictions for the masked sequence using top-k sampling."""
    inputs = tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
    # Sample from the top k probabilities
    sampled_indices = torch.multinomial(topk_probs.view(-1, top_k), 1)
    predicted_indices = torch.gather(topk_indices, 1, sampled_indices).view(-1)
    predicted_sequence = tokenizer.decode(predicted_indices)
    return predicted_sequence



def generate_dccm(pdb_path, output_path):
    """Generate the Dynamic Cross-Correlation Map (DCCM) from a PDB file using an R script."""
    r_script_path = './scripts/pdb_to_dccm.r'
    command = ['Rscript', r_script_path, pdb_path, output_path, 'all']
    subprocess.run(command, check=True)

def load_antipasti_model(checkpoint_path, input_shape):
    """Load the trained Antipasti model from a checkpoint."""
    model, _, _, _, _ = load_checkpoint(checkpoint_path, input_shape)
    model.eval()
    return model

def predict_binding_affinity(model, dccm):
    """Predict binding affinity from a DCCM map using the Antipasti model."""
    test_sample = torch.from_numpy(dccm.reshape(1, 1, model.input_shape, model.input_shape).astype(np.float32))
    output = model(test_sample)[0].detach().numpy()[0, 0]
    return 10 ** output

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python cdr_gen.py <antibody_sequence> <antigen_sequence> <number_of_candidates>")
        sys.exit(1)

    
    model_dir = './model_checkpoints'
    model, tokenizer = load_model(model_dir)
    antibody_sequence, antigen_sequence = sys.argv[1], sys.argv[2]
    num_candidates = int(sys.argv[3])
    candidates = []

   
    original_full_sequence = antibody_sequence + '[SEP]' + antigen_sequence
    original_pdb_path = generate_structure(original_full_sequence)
    original_dccm_path = './output_structures/original_dccm.npy'
    generate_dccm(original_pdb_path, original_dccm_path)
    original_dccm = np.load(original_dccm_path)
    antipasti_model = load_antipasti_model('./antipasti/checkpoints/full_ags_all_modes/model_epochs_1044_modes_all_pool_1_filters_4_size_4.p', 281)
    original_affinity = predict_binding_affinity(antipasti_model, original_dccm)

    # Generate and screen new sequences
    while len(candidates) < num_candidates:
        predicted_sequence = predict_masked_sequence(model, tokenizer, original_full_sequence)
        pdb_file_path = generate_structure(predicted_sequence)
        dccm_output_path = './output_structures/predicted_dccm.npy'
        generate_dccm(pdb_file_path, dccm_output_path)
        predicted_dccm = np.load(dccm_output_path)
        predicted_affinity = predict_binding_affinity(antipasti_model, predicted_dccm)

        if predicted_affinity > original_affinity:
            candidates.append(predicted_sequence)
            print(f"Found improved candidate: {predicted_sequence} with affinity {predicted_affinity}")
        else:
            print("Candidate discarded due to lower affinity.")

    print(f"Total candidates found: {len(candidates)}")
