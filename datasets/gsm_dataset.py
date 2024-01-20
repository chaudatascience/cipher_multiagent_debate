import numpy as np
from torch.utils.data import Dataset
import random
from models.utils import read_jsonl
import json


class GSM8K(Dataset):
    def __init__(self, input_path: str, n_ques: int, seed: int = 2023):
        random.seed(seed)
        np.random.seed(seed)

        all_questions = read_jsonl(input_path)
        random.shuffle(all_questions)
        self.questions = all_questions[:n_ques]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx]
