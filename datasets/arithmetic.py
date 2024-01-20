from typing import List, Dict
import numpy as np
from torch.utils.data import Dataset
import random
from models.utils import read_jsonl
import json
import os


def generate_one_data(ques_seed: int) -> Dict:
    np.random.seed(ques_seed)
    a, b, c, d, e, f = np.random.randint(0, 30, size=6)
    question = """What is the result of {}+{}*{}+{}-{}*{}?""".format(a, b, c, d, e, f)
    gt = a + b * c + d - e * f

    res = {"question": question, "answer": str(gt)}
    return res


def generate_data(seed: int, n_questions: int) -> List:
    data_list = []
    for ques_idx in range(n_questions):
        data = generate_one_data(ques_seed=seed + ques_idx)

        data_list.append(data)
    return data_list


def run_generate_dataset(
    seed: int = 10, n_questions: int = 200, output_dir: str = "data/arithmetic"
):
    data_list = generate_data(seed=seed, n_questions=n_questions)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"test_seed{seed}.jsonl")
    with open(output_path, "w") as f:
        for i, data in enumerate(data_list):
            json.dump(data, f)

            ## write "\n" except the last line
            if i < len(data_list) - 1:
                f.write("\n")


class Arithmetic(Dataset):
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


if __name__ == "__main__":
    run_generate_dataset(23, 200)
