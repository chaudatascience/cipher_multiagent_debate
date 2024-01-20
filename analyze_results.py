import json
import re
from typing import Dict, Optional
from evaluations.eval_number_accuracy import parse_agent_answer
import glob
import pandas as pd
from models.utils import duplicate_temp


## display all columns
pd.set_option("display.max_columns", None)
from evaluations.eval_utils import most_frequent


from fraction import Fraction
import sys

MAX_INT = sys.maxsize


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata

        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion):
    """
    This evaluation code is used for WizardMath
    Source code for WizardMath repo
    """
    text = completion.split("The answer is: ")
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r"[\-+]?\d*[\.,/]?\d+", extract_ans)
        if match:
            if "/" in match.group():
                denominator = match.group().split("/")[1]
                numerator = match.group().split("/")[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == "0":
                        return round(float(numerator.replace(",", "")))
                    else:
                        frac = Fraction(match.group().replace(",", ""))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(",", "")) == float("inf"):
                    return None
                return round(float(match.group().replace(",", "")))
        else:
            return None
    else:
        return None


def parse_abcd(detailed_ans: str) -> str:
    match_1 = re.search(r"Answer: (\w)", detailed_ans)
    match_2 = re.search(r"(correct answer is|the answer is|The answer is) (\w)", detailed_ans)
    if match_1:
        option_choice = match_1.group(1)
    elif match_2:
        option_choice = match_2.group(2)
    else:
        if detailed_ans != "":
            option_choice = detailed_ans[-1].upper()
        else:
            ## random A,B,C,D
            option_choice = "-"
    return option_choice


def agent_change_their_mind(output_path: str, n_rounds: int, agent_idx: int, n_questions: int):
    if "gsm8k" in output_path:
        if "wizardmath" in output_path.lower():
            parse_answer = extract_answer_number
        else:
            parse_answer = parse_agent_answer
        parse_gt = parse_agent_answer  ## check on minus numbers, e.g., #### -3
    elif "mmlu" in output_path:
        parse_answer = parse_abcd
        parse_gt = parse_abcd
    elif "arithmetic" in output_path:
        parse_answer = parse_agent_answer
        parse_gt = int
    else:
        raise NotImplementedError(f"Not supported for dataset: {output_path}")

    response_dict = json.load(open(output_path, "r"))

    questions = list(response_dict.keys())

    ### find n_agents, n_rounds
    *agent_answers_tmp, gt_tmp = response_dict[questions[0]]
    n_agents = len(agent_answers_tmp)

    detailed_dict = {"question": []}
    detailed_dict.update({f"round_{i}": [] for i in range(n_rounds)})
    detailed_dict.update({"gt": []})

    if n_questions is not None:
        print(f"only evaluate on the first {n_questions} questions!")
        questions = questions[:n_questions]

    for question in questions:
        *agent_answers_list, gt_answer = response_dict[question]
        gt = parse_gt(gt_answer)
        agent_answers = agent_answers_list[agent_idx]

        detailed_dict["gt"].append(gt)
        detailed_dict["question"].append(question)

        for r in range(n_rounds):
            agent_final_ans = parse_answer(agent_answers[r])
            detailed_dict[f"round_{r}"].append(agent_final_ans)

    detailed_df = pd.DataFrame(detailed_dict)
    temps = parse_temp_trial(output_path, n_rounds, n_agents, return_temp_only=True)
    temps = duplicate_temp(n_rounds, n_agents, temps)

    summary_dict = {"agent": [f"agent {agent_idx} (temp={temps[agent_idx*n_rounds]})"]}
    summary_dict.update({f"round_{i} acc(%)": [] for i in range(n_rounds)})
    summary_dict.update({f"change round {r} -> {r+1}": [] for r in range(n_rounds - 1)})
    summary_dict.update({f"change round {0} -> {n_rounds-1}": []})

    for r in range(n_rounds):
        acc = sum(detailed_df[f"round_{r}"] == detailed_df["gt"]) / len(detailed_df) * 100
        summary_dict[f"round_{r} acc(%)"] = acc  ## type: ignore

        if r < n_rounds - 1:
            prev_round = r
            next_round = r + 1
        else:
            prev_round = 0
            next_round = n_rounds - 1
        change = f'{sum(detailed_df[f"round_{prev_round}"] != detailed_df[f"round_{next_round}"])}/{len(detailed_df)}'
        summary_dict[f"change round {prev_round} -> {next_round}"] = change  ## type: ignore
    summary_df = pd.DataFrame(summary_dict, index=[0])
    return detailed_df, summary_df


def parse_temp_trial(output_path: str, n_rounds: int, n_agents: int, return_temp_only: bool = False):
    first, _ = output_path.split("temp")[:2]
    trial = first.split("/")[-1].split("_")[0].replace("id", "")

    if trial != "":
        trial = int(trial)
    if "_topp" in output_path:
        all_temps = output_path.split("temp")[-1].split("_topp")[0].split("-")
    else:
        all_temps = output_path.split("temp")[-1].split("_nques")[0].split("-")
    if len(all_temps) == n_agents * n_rounds:
        temps = [float(all_temps[agent_idx * n_rounds]) for agent_idx in range(n_agents)]
    elif len(all_temps) == n_agents:
        temps = [float(x) for x in all_temps]
    elif n_agents == 1:
        temps = [float(x) for x in all_temps]
    else:
        raise ValueError("Check again on temperatures!")
    if return_temp_only:
        return temps
    else:
        return *temps, trial


def agents_change_their_minds_analysis(
    output_path: str, n_rounds: Optional[int], n_questions: Optional[int] = None
) -> Dict:
    if "gsm8k" in output_path:
        dataset = "gsm8k"
    elif "mmlu" in output_path:
        dataset = "mmlu"
    elif "arithmetic" in output_path:
        dataset = "arithmetic"
    else:
        raise NotImplementedError(f"Not supported for dataset: {output_path}")

    if "human" in output_path:
        human_or_vector = "Human"
    else:
        human_or_vector = "Vector"

    response_dict = json.load(open(output_path, "r"))

    questions = list(response_dict.keys())

    ### find n_agents, n_rounds
    *agent_answers_tmp, _ = response_dict[questions[0]]
    n_agents = len(agent_answers_tmp)
    n_rounds_data = len(agent_answers_tmp[0])

    if n_rounds is None:
        n_rounds = n_rounds_data
    else:
        n_rounds = min(n_rounds, n_rounds_data)

    temperatures = parse_temp_trial(output_path, n_rounds, n_agents, return_temp_only=True)
    ## idx of min temperatures
    idx_min_temps = temperatures.index(min(temperatures))

    summary_df_list = []
    detailed_df_list = []
    for agent_idx in range(n_agents):
        detailed_df, summary_df = agent_change_their_mind(output_path, n_rounds, agent_idx, n_questions)

        gt_df = detailed_df[["gt"]]
        detailed_df_temp = detailed_df[["question"]]
        detailed_df = detailed_df.drop(columns=["question", "gt"])
        detailed_df.columns = [f"{col} (agent {agent_idx})" for col in detailed_df.columns]

        detailed_df_list.append(detailed_df)
        summary_df_list.append(summary_df)

    detailed_df_list.append(gt_df)
    detailed_df_list.append(detailed_df_temp)

    all_detailed_df = pd.concat(detailed_df_list, axis=1)
    res = {}
    for r in range(n_rounds):
        all_detailed_df[f"aggree round_{r}"] = all_detailed_df[
            [f"round_{r} (agent {i})" for i in range(n_agents)]
        ].apply(lambda x: len(set(x)) == 1, axis=1)

        res[f"aggree round_{r}(%)"] = all_detailed_df[f"aggree round_{r}"].mean() * 100

        all_detailed_df[f"majority vote round_{r}"] = all_detailed_df[
            [f"round_{r} (agent {i})" for i in range(n_agents)]
        ].apply(lambda x: most_frequent(list(x)), axis=1)

        res[f"majority vote r_{r} acc"] = (
            sum(all_detailed_df[f"majority vote round_{r}"] == all_detailed_df["gt"])
            / len(all_detailed_df)
            * 100
        )
        res[f"max_acc_r_{r}"] = max(
            [
                sum(all_detailed_df[f"round_{r} (agent {i})"] == all_detailed_df["gt"])
                / len(all_detailed_df)
                * 100
                for i in range(n_agents)
            ]
        )

    pattern = r"round_\d+ \(agent \d+\)"
    all_rounds = [col for col in all_detailed_df.columns if re.search(pattern, col)]

    debate_rounds = [col for col in all_rounds if "round_0" not in col]
    last_rounds = [col for col in all_rounds if f"round_{n_rounds-1}" in col]
    all_detailed_df["major_debate"] = all_detailed_df[debate_rounds].apply(
        lambda x: most_frequent(list(x)), axis=1
    )
    major_debate_acc = (
        sum(all_detailed_df["major_debate"] == all_detailed_df["gt"]) / len(all_detailed_df) * 100
    )

    all_detailed_df["major_last_round"] = all_detailed_df[last_rounds].apply(
        lambda x: most_frequent(list(x)), axis=1
    )
    major_last_round = (
        sum(all_detailed_df["major_last_round"] == all_detailed_df["gt"]) / len(all_detailed_df) * 100
    )

    all_detailed_df["major_all"] = all_detailed_df[all_rounds].apply(lambda x: most_frequent(list(x)), axis=1)
    major_all_acc = sum(all_detailed_df["major_all"] == all_detailed_df["gt"]) / len(all_detailed_df) * 100

    all_summary_df = pd.concat(summary_df_list, ignore_index=True)

    agent_1_idx = 1 if n_agents >= 2 else 0
    agent_lowest_temp_last_round = all_summary_df[f"round_{n_rounds-1} acc(%)"][idx_min_temps]
    ## agent with lowest temperature

    max_acc_last_round = all_summary_df[f"round_{n_rounds-1} acc(%)"].max()
    print(
        f"-------------------[{human_or_vector}] {n_agents} agents: Max acc on {dataset} of the last round: {max_acc_last_round:.2f}%"
    )
    print(f"major debate rounds:", major_debate_acc)
    print(f"major all rounds:", major_all_acc)
    print(f"major_last_round", major_last_round)

    n_rows, n_cols = all_summary_df.shape
    majority_vote_accs = [res[f"majority vote r_{r} acc"] for r in range(n_rounds)]
    agree_rounds = [res[f"aggree round_{r}(%)"] for r in range(n_rounds)]
    max_acc_rounds = [res[f"max_acc_r_{r}"] for r in range(n_rounds)]

    all_summary_df.loc[n_rows + 1] = ["agreement"] + agree_rounds + ["-"] * (n_cols - 1 - n_rounds)

    all_summary_df.loc[n_rows] = [f"majority vote"] + majority_vote_accs + ["-"] * (n_cols - 1 - n_rounds)

    all_summary_df.loc[n_rows + 2] = ["max acc"] + max_acc_rounds + ["-"] * (n_cols - 1 - n_rounds)

    res_dict = {}
    res_dict["detailed_df"] = all_detailed_df
    res_dict["summary_df"] = all_summary_df
    *temps, trial_id = parse_temp_trial(output_path, n_rounds, n_agents)
    res_dict["major_debate_acc"] = (temps, (round(major_debate_acc, 2), trial_id))
    res_dict["agent_lowest_temp_last_round"] = (
        temps,
        (round(agent_lowest_temp_last_round, 2), trial_id),
    )
    return res_dict


def analyse_results_whole_folder(
    dataset: str,
    sub_dataset: str,
    model: str,
    type_: str,
    verbose: bool,
    n_rounds: Optional[int] = None,
    n_questions: Optional[int] = None,
    contour_plots: bool = False,
):
    """
    type_: human or vector, or both
    model: num_debaters + model name, such as 3Llama-2-70b-hf,
            or path to the model (e.g., output/log_2023-Jul-26/id15890114_1Llama-2-70b-hf_gsm8k_test_c...)
    """
    if "/output/" in model:
        model = (
            model.split("output/")[-1] + "output/"
        )  ## keep only "output/log_..." if passing absolute path of the model
    if ".json" in model:  ## use path to evaluate on a file
        filter_out = []
    else:
        filter_out = ["majority_vote", "testing", "_r8_"]

    all_files = glob.glob(f"output/*/*.json")
    all_files = [f for f in all_files if dataset in f and sub_dataset in f and model in f]

    if contour_plots:
        all_files = [
            f
            for f in all_files
            if (
                "2Llama-2-70b-hf" in f
                and "llama_65B" not in f
                and "expert" not in f
                and "id_" not in f
                and "Llama-2-70b-chat-hf" not in f
                and "nques2_" not in f
                and "nques8_" not in f
            )
        ]

    if type_ == "human":
        filter_out.append("_vec")
    elif type_ == "vector":
        filter_out.append("human")

    all_files = [f for f in all_files if not any([x in f for x in filter_out])]
    all_files = sorted(all_files, key=lambda x: x.split("temp")[-1])

    ## ignore testing files: file names that don't have a trial ID will be discarded.
    if not ".json" in model:  ## use path to evaluate on a file
        all_files = [f for f in all_files if "id_" not in f]
        all_files = [f for f in all_files if "testing_" not in f]
        # all_files = [f for f in all_files if "_r8_" not in f]

    all_files = [f for f in all_files if "_nsols1_" in f]  ## filter out majority vote

    result_human_debate_major = []
    result_vector_debate_major = []

    result_human_agent_lowest_temp_last_round = []
    result_vector_agent_lowest_temp_last_round = []

    for f_i, f in enumerate(all_files, 1):
        print(f"{f_i}/{len(all_files)}")
        print(f)
        res_dict = agents_change_their_minds_analysis(f, n_rounds, n_questions)
        print(res_dict["summary_df"])
        if "human" in f:
            result_human_debate_major.append(res_dict["major_debate_acc"])
            result_human_agent_lowest_temp_last_round.append(res_dict["agent_lowest_temp_last_round"])
        elif "_vec" in f:
            result_vector_debate_major.append(res_dict["major_debate_acc"])
            result_vector_agent_lowest_temp_last_round.append(res_dict["agent_lowest_temp_last_round"])

        if verbose:
            res_dict["detailed_df"].to_csv("output/tmp.csv", index=True)
            print("wrote tmp.csv file in output/tmp.csv!")

        print("-------")
    result_human_debate_major = sorted(result_human_debate_major, key=lambda x: x[1])
    result_vector_debate_major = sorted(result_vector_debate_major, key=lambda x: x[1])

    result_human_agent_lowest_temp_last_round = sorted(
        result_human_agent_lowest_temp_last_round, key=lambda x: x[1]
    )
    result_vector_agent_lowest_temp_last_round = sorted(
        result_vector_agent_lowest_temp_last_round, key=lambda x: x[1]
    )

    return (
        result_human_debate_major,
        result_human_agent_lowest_temp_last_round,
        result_vector_debate_major,
        result_vector_agent_lowest_temp_last_round,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze results.")
    parser.add_argument("-d", "--dataset", type=str, default="", help="Name of dataset")
    parser.add_argument("-s", "--sub_dataset", type=str, default="", help="Name of sub-dataset")
    parser.add_argument("-m", "--model", type=str, default="3Llama-2-70b-hf", help="Name of model")
    parser.add_argument("-r", "--n_rounds", type=int, default=None, help="Number of rounds")
    parser.add_argument(
        "-c", "--contour_plots", action="store_true", help="Filter points to plot contour plots"
    )

    parser.add_argument("-n", "--n_questions", type=int, default=None, help="num questions to evaluate")
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="both",
        help="Type of analysis",
        choices=["human", "vector", "both"],
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print more info")

    args = parser.parse_args()

    (
        human_debate_major,
        human_agent_lowest_temp_last_round,
        vector_debate_major,
        vector_agent_lowest_temp_last_round,
    ) = analyse_results_whole_folder(
        dataset=args.dataset,
        sub_dataset=args.sub_dataset,
        model=args.model,
        type_=args.type,
        verbose=args.verbose,
        n_rounds=args.n_rounds,
        n_questions=args.n_questions,
        contour_plots=args.contour_plots,
    )
    print("_" * 30)

    print(args)

    print("The results are in the following order: [temperatures], (accuracy, job_id)")
    print(
        f"\nhuman majority voting of debate rounds ({len(human_debate_major)} runs):",
        human_debate_major,
        "\n",
    )

    print(
        f"vector majority voting of debate rounds ({len(vector_debate_major)} runs):",
        vector_debate_major,
        "\n\n",
    )

    print("_" * 30)
    #### human_agent_lowest_temp_last_round  last round
    print(
        f"human_agent_lowest_temp_last_round ({len(human_agent_lowest_temp_last_round)} runs):",
        human_agent_lowest_temp_last_round,
        "\n",
    )
    print(
        f"vector_agent_lowest_temp_last_round ({len(vector_agent_lowest_temp_last_round)} runs):",
        vector_agent_lowest_temp_last_round,
        "\n",
    )


if __name__ == "__main__":
    main()
