import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from abnumber import Chain

# Load model and tokenizer
model_name = "alchemab/antiberta2"
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load dataset
dataset_name = "ASIRI25/cdrgen"
raw_datasets = load_dataset(dataset_name)

def mask_cdr3(antibody):
    chain = Chain(antibody, scheme='chothia')
    cdr3 = chain.get_cdr('CDR3')
    if cdr3:
        start, end = cdr3.start_pos-1, cdr3.end_pos  # -1 because abnumber is 1-based index
        return antibody[:start] + '[MASK]' * (end - start) + antibody[end:], (start, end)
    return antibody, None

class AntibodyAntigenDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        sequence = self.entries[idx]['sequence']
        antibody, antigen = sequence.split('[SEP]')
        masked_antibody, cdr3_range = mask_cdr3(antibody)
        if cdr3_range:
            masked_sequence = masked_antibody + '[SEP]' + antigen
            inputs = tokenizer(masked_sequence, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
            labels = tokenizer(sequence, return_tensors='pt', padding='max_length', max_length=512, truncation=True)['input_ids']
            return inputs.input_ids.squeeze(0), inputs.attention_mask.squeeze(0), labels.squeeze(0)
        else:
            return None

processed_entries = [entry for entry in raw_datasets['train'] if mask_cdr3(entry['sequence'].split('[SEP]')[0])[1]]
dataset = AntibodyAntigenDataset(processed_entries)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Optimizer and training loop
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
model.train()
for epoch in range(3):
    for batch in loader:
        if batch is None:
            continue
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        predictions = outputs.logits

        # Custom loss: calculate loss only for masked CDR3 region
        loss = F.cross_entropy(predictions.view(-1, tokenizer.vocab_size), labels.view(-1), reduction='none')
        loss_mask = (labels == tokenizer.mask_token_id).view(-1)
        loss = (loss * loss_mask).sum() / loss_mask.sum()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch}, Loss: {loss.item()}")


model.save_pretrained('./model_checkpoints')
tokenizer.save_pretrained('./model_checkpoints')
