import numpy as np
from torch.utils.data import Dataset
import random
import pandas as pd


class MMLU(Dataset):
    def __init__(self, input_path: str, n_ques: int, seed: int = 2023):
        random.seed(seed)
        np.random.seed(seed)

        all_questions = pd.read_csv(input_path, header=None)
        all_questions = all_questions.sample(frac=1, random_state=seed).reset_index(drop=True)
        self.questions = all_questions[:n_ques]

    def __len__(self):
        return len(self.questions)

    def parse_question_answer(self, idx):
        ques = self.questions.iloc[idx, 0]
        a = self.questions.iloc[idx, 1]
        b = self.questions.iloc[idx, 2]
        c = self.questions.iloc[idx, 3]
        d = self.questions.iloc[idx, 4]

        full_ques = "{}\nA. {}\nB. {}\nC. {}\nD. {}".format(str(ques).strip(), a, b, c, d)

        gt = self.questions.iloc[idx, 5]

        return full_ques, gt

    def __getitem__(self, idx):
        """
        return a dict obj with 2 keys: `question`, `answer`
        """

        full_ques, gt = self.parse_question_answer(idx)
        return {"question": full_ques, "answer": gt}
