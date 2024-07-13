from colabfold.batch import run_alphafold
import os

def generate_structure(sequence):
    config = {
        "max_recycles": 20,
        "early_stop_tolerance": 0.5,
        "template_mode": "pdb100"
    }
    result = run_alphafold(
        sequence=sequence,
        model_type="AlphaFold2_multimer",  # Use multimer model if considering interactions
        **config
    )
    pdb_data = result['result'][0]['model_1']
    
    # Ensure output directory exists
    output_dir = './output_structures'
    os.makedirs(output_dir, exist_ok=True)
    
    pdb_path = os.path.join(output_dir, 'predicted_structure.pdb')
    with open(pdb_path, 'w') as file:
        file.write(pdb_data)
    
    return pdb_path

if __name__ == "__main__":
    sample_sequence = 'YOUR_COMBINED_SEQUENCE_HERE'
    pdb_file_path = generate_structure(sample_sequence)
    print(f"Generated PDB file saved at: {pdb_file_path}")
