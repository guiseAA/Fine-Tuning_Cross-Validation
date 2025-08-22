# src/dataset.py
import torch
from torch.utils.data import Dataset

class CodeAffinityDataset(Dataset):
    def __init__(self, Method_Code, Snipped_Code, affinities, tokenizer, max_length=512):
        self.Method_Code = Method_Code
        self.Snipped_Code = Snipped_Code
        self.affinities = affinities
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.Method_Code)

    def __getitem__(self, idx):
        Method_method_pair = f"Method: {self.Method_Code[idx]} Snipped: {self.Snipped_Code[idx]}"
        encoding = self.tokenizer(Method_method_pair, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        label = torch.tensor(self.affinities[idx], dtype=torch.float)  # Regressión para afinidad
        return {'input_ids': encoding['input_ids'].squeeze(0), 'attention_mask': encoding['attention_mask'].squeeze(0), 'labels': label}
