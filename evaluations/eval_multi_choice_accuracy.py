from models.utils import get_model_path
from transformers import LlamaTokenizer

import torch
from typing import List, Union
from evaluations.eval_utils import most_frequent
import random
import re
import json
import yaml

pprint = lambda x: print(json.dumps(x, indent=2, ensure_ascii=False))


def get_tokenizer_and_emb_table(model: str):
    with open("config.yml", "r") as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)

    emb_path = yaml_config["emb_path"].format(model=model)
    model_path = get_model_path(model)

    if model.startswith("llama"):
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        emb_table = torch.load(emb_path, map_location=torch.device("cpu")).float()
    else:
        raise NotImplementedError
    return tokenizer, emb_table


def compute_accuracy_for_multi_choice_answer(
    gt: str, pred_solutions: Union[List[str], str], tokenizer, emb_table, emb_abcd
):
    def extract_final_ans(ans: str):
        if ans == "":
            print("----Warning, ans is empty!!!")
            ## random from A, B, C, D:
            option_choice = random.choice(["A", "B", "C", "D"])
            return option_choice

        match = re.search(r"Answer: (\w)", ans)
        if match:
            option_choice = match.group(1)
        else:
            option_choice = ans[-1].upper()
        mapper = {"0": "A", "1": "B", "2": "C", "3": "D"}

        if option_choice in list(mapper.values()):
            return option_choice
        else:
            token_id = tokenizer.encode(option_choice, add_special_tokens=False)[0]
            emb = emb_table[token_id]

            ## find closest distance from emb to emb_abcd
            dist = torch.cdist(emb.unsqueeze(0), emb_abcd, p=2)
            min_dist, min_idx = torch.min(dist, dim=1)
            nearest_option_choice = mapper[str(min_idx.item())]

            return nearest_option_choice

    gt = gt.upper()

    if isinstance(pred_solutions, str):
        answer = extract_final_ans(pred_solutions)
    else:
        answer_list = [extract_final_ans(pred_sol) for pred_sol in pred_solutions]
        answer = most_frequent(answer_list)
    return answer == gt
