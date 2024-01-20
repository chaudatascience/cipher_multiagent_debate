from evaluations.eval_utils import most_frequent
from typing import Optional
import re


def solve_math_problems(input_str):
    input_str = input_str.replace(",", "")
    pattern = r"-?\d+\.?\d*"

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None


def parse_answer(input_str):
    pattern = r"\{(-?[0-9.,$]*)\}"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break

    return solution


def parse_agent_answer(answer_str: str):
    """parse Answer from a string
    e.g., "games after 3 years.\nAnswer: 144``` => 144
    """
    pattern = r"Answer: (-?\d+)"  #
    pattern_float = r"Answer: (-?\d+\.\d+)"  #

    input_str = answer_str
    for c in ["$", ",", "€", '"']:
        input_str = input_str.replace(c, "")

    ## Find the matching pattern in the string
    match = re.search(pattern, input_str)
    match_2 = re.search(pattern_float, input_str)
    number = None
    dummy_ans = -99999

    if match_2:
        # Extract the number from the matched pattern
        number = float(match_2.group(1))
    elif match:
        number = float(match.group(1))
    else:
        number = parse_answer(input_str)
        if number is None:
            number = solve_math_problems(input_str)
        if number is None:
            number = dummy_ans

    try:
        number = float(number)
        return number
    except:
        print("Warning: can't parse the answer")
        return dummy_ans


def compute_accuracy_for_number_answer(gt, pred_solutions, **kargs) -> Optional[int]:
    gt_ans = solve_math_problems(gt)
    if gt_ans is None:
        return None

    if isinstance(pred_solutions, list):
        pred_answer_list = []
        for pred_sol in pred_solutions:
            pred_answer = parse_agent_answer(pred_sol)
            pred_answer_list.append(float(pred_answer))

        pred_answer = most_frequent(pred_answer_list)

    else:
        pred_answer = parse_agent_answer(pred_solutions)

    eps = 1e-5
    if abs(float(gt_ans) - pred_answer) < eps:
        return 1
    else:
        return 0
