import os
import json
from os.path import join as os_join
from tqdm import tqdm
import time
import argparse
from argparse import Namespace
from models.agent import Agent
from models.utils import *
from typing import List, Dict, Optional
from collections import OrderedDict
from datasets import *
from torch.utils.data import DataLoader
from copy import deepcopy
from collections import defaultdict
import numpy as np
import random
import os
import yaml

from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from bayes_opt.logger import JSONLogger

torch.manual_seed(2023)
from torch.utils.data import random_split
import ray
from analyze_results import agents_change_their_minds_analysis
from models.utils import duplicate_temp

log_file_detailed = None
pprint = lambda x: print(json.dumps(x, indent=2, ensure_ascii=False))


def set_seeds(seed=2022):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    ########################### Arguments ###########################
    parser = argparse.ArgumentParser(description="Multiagent debate - version 1.0")
    parser.add_argument(
        "--debaters",
        type=type_list("str"),
        help="List of debaters, choices: falcon-7b-instruct, falcon-40b-instruct,\
        llama_7B, llama_13B, llama_30B, llama_65B, llama_7B_expert, llama_65B_expert\
        Llama-2-70b-hf, Llama-2-70b-chat-hf,  Llama-2-70b-hf_expert, Llama-2-70b-chat-hf_expert, Llama-2-70b-hf_dummy_expert",
        default=["llama_7B", "llama_7B"],  ##["falcon-40b-instruct", "llama_65B"],
    )

    ## TODO: add judge to finalize the final answer.
    parser.add_argument("--judge", type=str, help="same choices as debaters", default=None)

    parser.add_argument("--n_questions", type=int, help="Number of questions", default=None)

    parser.add_argument("--n_ray_actors", type=int, help="Number parallel actors", default=2)

    parser.add_argument("--n_gpus_per_actor", type=int, help="Number GPUs per actors", default=4)

    parser.add_argument("--partial_thres", type=float, default=0.85)

    parser.add_argument(
        "--partial_entropy_over_max",
        action="store_true",
        default=False,
        help="use the entropy over max. Default: use max",
    )

    parser.add_argument(
        "--partial_cipher_when_confident",
        action="store_true",
        default=False,
        help="use the weighted avg when the model is confident, otherwise use the best token",
    )

    parser.add_argument(
        "--partial_cipher_when_not_confident",
        action="store_true",
        default=False,
        help="use the weighted avg when the model is NOT confident, otherwise use the best token",
    )

    parser.add_argument(
        "--initial_prompt_paths",
        type=type_list("str"),
        help="Path to the initial prompt file",
        default=["prompts/gsm8k/init_question_3shot_v3.txt"],
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "mmlu", "arithmetic", "biography"],
    )

    parser.add_argument(
        "--debate_prompt_paths",
        type=type_list("str"),
        help="Path to the debate prompt file",
        default=["prompts/gsm8k/debate_3shot_llama_2debaters.txt"],
    )

    parser.add_argument(
        "--temperatures",
        type=type_list("float"),
        help=f"m * r numbers, where m is #debaters, r is #rounds: model1_r1, model1_r2, model1_r3, model2_r1, model2_r2, model2_r3",
        default=[],
    )

    parser.add_argument(
        "--top_ps",
        type=type_list("float"),
        help="Top p value for the models at each round",
        default=[0.85],
    )

    parser.add_argument("--max_new_tokens", type=int, help="Maximum number of new tokens", default=400)

    parser.add_argument("--load_in_8bit", action="store_true", help="Load in 8bit", default=False)
    parser.add_argument(
        "--positional_bias",
        action="store_true",
        help="swap the order of other agent and current agent",
        default=False,
    )

    parser.add_argument("--use_bayesian", action="store_true", default=False)

    parser.add_argument(
        "--no_early_stop",
        action="store_true",
        help="turn off early_stop. early_stop to truncate output emb answer when using vector language",
        default=False,
    )

    parser.add_argument(
        "--no_convert_ans_choice",
        action="store_true",
        help="if we don't  want to convert the last token to A,B,C,D",
        default=False,
    )

    # Default 1. If more than 1, we will sample `sampling` answers for each question,
    # and use majority vote to have a final answer. it stops after round 0.
    parser.add_argument("--n_sols_each_ques", type=int, help="Sampling method", default=1)

    # This is used when `sampling` is set to 1
    parser.add_argument("--n_rounds", type=int, help="Number of rounds", default=3)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data_sample_seed", type=int, default=2023)
    parser.add_argument("--temperature_min", type=float, default=0.01)
    parser.add_argument("--temperature_max", type=float, default=1.0)

    parser.add_argument("--debug", action="store_true", help="Debug mode", default=False)

    parser.add_argument("--root_dir", type=str, default="./")
    parser.add_argument("--bayesian_json_log", type=str, default="")
    parser.add_argument("--bayesian_num_init_points", type=int, default=2)
    parser.add_argument(
        "--bayesian_criterion",
        type=str,
        default="major_last_round",
        choices=["major_last_round", "major_debate_acc", "major_all_rounds"],
    )

    parser.add_argument("--data_path", type=str, default="data/gsm/test_gsm8k_full.jsonl")

    parser.add_argument("-b", "--batch_size", type=int, default=16)

    parser.add_argument("-v", "--vector_language", action="store_true", default=False)

    parser.add_argument(
        "-p",
        "--num_points",
        type=int,
        default=1,
        help="num experiments to run random search",
    )

    parser.add_argument(
        "--point_path",
        type=str,
        default="",
        help="path for pre-defined points that run random search on first",
    )

    ## add top_p_emb
    parser.add_argument("--top_p_emb", type=float, default=1.0, help=">=1 mean not using")

    parser.add_argument("--top_k_emb", type=int, default=-1, help="-1 mean not using")

    parser.add_argument("--l2_norm", action="store_true", default=False)

    args = parser.parse_args()
    return args


def run_majority_vote(
    n_sols: int,
    questions: List[str],
    gts: List[str],
    agent_name: str,
    agent: Agent,
    temperature: float,
    top_p: float,
    top_p_emb: float,
    top_k_emb: int,
    l2_norm: bool = False,
    early_stop: bool = True,  ## this is used with vector only. i.e., ignored in this function
    use_ray: bool = False,
):
    sol_history_batch_ans_token = defaultdict(list)

    print_out("================================================")
    for sol_i in range(n_sols):
        if not use_ray:
            ## give a direct answer
            answer_batch, answer_batch_ans_token, prompt_batch = agent.give_first_solutions(
                questions=questions,
                temperature=temperature,
                top_p=top_p,
                vector_language=False,
                top_p_emb=top_p_emb,
                l2_norm=l2_norm,
                top_k_emb=top_k_emb,
                early_stop=early_stop,
            )
        else:
            handler = agent.give_first_solutions.remote(
                questions=questions,
                temperature=temperature,
                top_p=top_p,
                vector_language=False,
                top_p_emb=top_p_emb,
                l2_norm=l2_norm,
                top_k_emb=top_k_emb,
                early_stop=early_stop,
                convert_to_cpu=True,
            )
            res_tmp = ray.get(handler)
            answer_batch, answer_batch_ans_token, prompt_batch = res_tmp
        if sol_i == 0:
            print_out(f"\n========={agent_name}: prompt sol: \n{prompt_batch}=========")

        print_out(f"\n========={agent_name}: answer sol {sol_i+1}: \n{answer_batch}=========")
        for ques, answer_ans_token in zip(questions, answer_batch_ans_token):
            sol_history_batch_ans_token[ques].append(answer_ans_token)

    res = {}
    for ques, gt in zip(questions, gts):
        temp = {ques: (sol_history_batch_ans_token[ques], gt)}
        res.update(temp)
    return res


def run_multiagent_debate(
    n_rounds: int,
    questions: List[str],
    gts: List[str],
    debaters: List[str],
    agents: Dict[str, Agent],
    temperatures_dict: Dict,
    top_ps: List,
    vector_language: bool,
    top_p_emb: float,
    top_k_emb: int,
    l2_norm: bool,
    early_stop: bool,
    using_ray: bool,
    llama1_vs_llama2: bool = False,
):
    def collect_answer(sol_history: Dict, gt: str):
        """
        sol_history: history of answers for 1 question
        """
        all_ans = []
        for agent_i, agent_name in enumerate(debaters):
            all_ans.append([sol_history[f"{agent_name}_{agent_i}"][i] for i in range(n_rounds)])
        res = {ques: (*all_ans, gt)}
        return res

    sol_history_batch = OrderedDict()
    other_sol_history_batch = OrderedDict()  ## for llama1_vs_llama2
    sol_history_batch_text = OrderedDict()
    for ques in questions:
        sol_history = {f"{agent_name}_{agent_i}": {} for agent_i, agent_name in enumerate(debaters)}
        sol_history_batch[ques] = sol_history
        other_sol_history_batch[ques] = deepcopy(sol_history)
        sol_history_batch_text[ques] = deepcopy(sol_history)

    print_out("================================================")
    for r in range(n_rounds):
        for agent_i, agent_name in enumerate(debaters):
            agent_temp = temperatures_dict[f"{agent_name}_{agent_i}"][r]
            clear_gpu_mem(verbose=True)
            agent_name_clean = agent_name.replace("_dummy", "").replace("_expert", "")
            agent = agents[agent_name_clean]
            if "expert" in agent_name:
                agent.use_expert_or_dummy_expert = True
                if "dummy" in agent_name:
                    ## random i in range(len(gts))
                    i = random.randint(1, len(gts) - 1)
                    gt_for_expert = gts[i:] + gts[:i]
                else:
                    gt_for_expert = gts
            else:
                agent.use_expert_or_dummy_expert = False
                gt_for_expert = None
            if r == 0:
                if using_ray:
                    ## Note on Ray: current issue, can not ray.get(handle) if handle contains a GPU tensor and the function doesn't have GPU left to use

                    ## give a direct answer
                    handler = agent.give_first_solutions.remote(
                        questions=questions,
                        temperature=agent_temp,
                        top_p=top_ps[r],
                        vector_language=vector_language,
                        top_p_emb=top_p_emb,
                        top_k_emb=top_k_emb,
                        l2_norm=l2_norm,
                        gt_for_expert=gt_for_expert,
                        early_stop=early_stop,
                        convert_to_cpu=True,
                    )

                    res_tmp = ray.get(handler)
                else:
                    res_tmp = agent.give_first_solutions(
                        questions=questions,
                        temperature=agent_temp,
                        top_p=top_ps[r],
                        vector_language=vector_language,
                        top_p_emb=top_p_emb,
                        top_k_emb=top_k_emb,
                        l2_norm=l2_norm,
                        gt_for_expert=gt_for_expert,
                        early_stop=early_stop,
                    )

            else:
                ## debate
                ## First, get answers from the previous round if we haven't
                if agent_i == 0:
                    prev_sols_batch = []
                    for ques in questions:
                        prev_sols = tuple(
                            sol_history_batch[ques][f"{name}_{i}"][r - 1] for i, name in enumerate(debaters)
                        )
                        prev_sols_batch.append(prev_sols)
                    if llama1_vs_llama2 and vector_language:
                        other_prev_sols_batch = []  ## for llama1_vs_llama2
                        for ques in questions:
                            other_prev_sols = tuple(
                                other_sol_history_batch[ques][f"{name}_{i}"][r - 1]
                                for i, name in enumerate(debaters)
                            )
                            other_prev_sols_batch.append(other_prev_sols)
                    else:
                        other_prev_sols_batch = None
                if using_ray:
                    handler = agent.give_debate_solutions.remote(
                        questions=questions,
                        prev_sols_batch=prev_sols_batch,
                        other_sols_batch=other_prev_sols_batch,
                        agent_index=agent_i,
                        temperature=agent_temp,
                        top_p=top_ps[r],
                        vector_language=vector_language,
                        top_p_emb=top_p_emb,
                        top_k_emb=top_k_emb,
                        l2_norm=l2_norm,
                        gt_for_expert=gt_for_expert,
                        early_stop=early_stop,
                        convert_to_cpu=True,
                    )
                    res_tmp = ray.get(handler)
                else:
                    res_tmp = agent.give_debate_solutions(
                        questions=questions,
                        prev_sols_batch=prev_sols_batch,
                        other_sols_batch=other_prev_sols_batch,
                        agent_index=agent_i,
                        temperature=agent_temp,
                        top_p=top_ps[r],
                        vector_language=vector_language,
                        top_p_emb=top_p_emb,
                        top_k_emb=top_k_emb,
                        l2_norm=l2_norm,
                        gt_for_expert=gt_for_expert,
                        early_stop=early_stop,
                    )

            if vector_language:
                answer_batch = res_tmp["emb"]
                other_answer_batch = res_tmp.get("other_emb", None)  ## List[tensor] or None
                prompt_batch = res_tmp["prompt"]
                answer_nearest_batch = res_tmp["nearest_neighbor_texts"]
            else:
                answer_batch, answer_batch_ans_token, prompt_batch = res_tmp
                other_answer_batch = None

            print_out(f"\n====={agent_name}_{agent_i}, temp={agent_temp}: prompt round {r}: =====")
            print_out(prompt_batch)

            if not vector_language:
                print_out(f"\n====={agent_name}_{agent_i}, temp={agent_temp}: answer round {r}: =====")
                print_out(answer_batch_ans_token)
            if vector_language:
                print_out(f"temperature={agent_temp}")
                print_out("--- nearest_neighbor_texts: ")
                print_out(res_tmp["nearest_neighbor_texts"])

                # print_out("--- human_readable_texts: ")
                # print_out(res_tmp["human_readable_texts"])

            ## log the answer of each agent
            answer_texts = answer_nearest_batch if vector_language else answer_batch_ans_token
            for ques, answer, answer_txt in zip(questions, answer_batch, answer_texts):
                sol_history_batch[ques][f"{agent_name}_{agent_i}"][
                    r
                ] = answer  ## original answer from model, use as debate history

                sol_history_batch_text[ques][f"{agent_name}_{agent_i}"][
                    r
                ] = answer_txt  ## store to file for evaluation

            if other_answer_batch is not None:
                for ques, other_answer in zip(questions, other_answer_batch):
                    other_sol_history_batch[ques][f"{agent_name}_{agent_i}"][r] = other_answer

    ## collect answers for all questions
    res = {}
    for ques, gt in zip(questions, gts):
        tmp = collect_answer(sol_history_batch_text[ques], gt)
        res.update(tmp)
    return res


def print_out(x):
    try:
        pprint(x)
    except:
        print(x)
    # try:
    #     with open(log_file_detailed, "a") as f:
    #         f.write(str(x.encode("utf-8").strip()) + "\n")
    # except Exception as e:
    #     pass


def debate_helper(args, dataloader, agents: Optional[Dict] = None):
    ## for ablation: LLaMA1 vs. LLaMA2
    if set(args.debaters) == set(["Llama-2-70b-hf", "llama_65B"]) or set(args.debaters) == set(
        ["Llama-2-7b-hf", "llama_7B"]
    ):
        other_agent_embedding = True
    else:
        other_agent_embedding = False  # False

    debaters = args.debaters
    n_debaters = len(debaters)
    n_ques = args.n_questions
    initial_prompt_paths = maybe_duplicate(args.initial_prompt_paths, n_debaters)
    debate_prompt_paths = maybe_duplicate(args.debate_prompt_paths, n_debaters)

    max_new_tokens = args.max_new_tokens
    load_in_8bit = args.load_in_8bit
    n_sols = args.n_sols_each_ques
    n_rounds = args.n_rounds
    debug = args.debug
    root_dir = args.root_dir
    data_path = args.data_path
    vector_language = args.vector_language
    top_p_emb = args.top_p_emb
    top_k_emb = args.top_k_emb
    l2_norm = args.l2_norm
    dataset_name = args.dataset
    no_convert_ans_choice = args.no_convert_ans_choice
    early_stop = not args.no_early_stop
    n_gpus_per_actor = args.n_gpus_per_actor
    n_ray_actors = args.n_ray_actors
    positional_bias = args.positional_bias
    partial_thres = args.partial_thres
    partial_entropy_over_max = args.partial_entropy_over_max

    # extract filename from `data_path`
    filename = data_path.split("/")[-1].split(".")[0]

    temperatures = maybe_duplicate(args.temperatures, n_rounds * n_debaters)
    top_ps = maybe_duplicate(args.top_ps, n_rounds)

    datetime_short = datetime_now("%Y-%b-%d")
    start = time.time()

    trial = os.environ.get("JOB_ID", "")
    temps_str = "-".join([str(round(float(x), 3)) for x in temperatures])
    # top_ps_str = "-".join([str(x) for x in top_ps])
    use_vector_language = "vec" if vector_language else "human"

    log_file_detailed_name = f"id{trial}_{n_debaters}{debaters[0]}{debaters[1]}_{dataset_name}_{filename}_{use_vector_language}_{args.seed}_nsols{n_sols}_r{n_rounds}_temp{temps_str}_nques{n_ques}.json"

    global log_file_detailed
    log_file_detailed = os_join(
        "output",
        "detailed_logs",
        log_file_detailed_name.replace(".json", ".log"),
    )

    print_out(f"JOB_ID = {trial}")

    ## Assert the debate prompt template and n_debaters are consistent
    if n_sols == 1:
        assert all(
            f"{n_debaters}debaters" in path for path in debate_prompt_paths
        ), "check again on the debate templates!"

    if vector_language:
        assert all(
            "vector_language" in path for path in debate_prompt_paths
        ), "check again on the debate templates!"
    else:
        assert all(
            "vector_language" not in path for path in debate_prompt_paths
        ), "check again on the debate templates!"

    if args.positional_bias:
        debate_prompt_paths_bias = []
        for prompt in debate_prompt_paths:
            if "bias" not in prompt:
                prompt = prompt.replace(".txt", "_bias.txt")
                debate_prompt_paths_bias.append(prompt)

        debate_prompt_paths = debate_prompt_paths_bias
        print("+++++++++++++++ auto add _bias to the prompt for bias ablation!")

    ## set up agentss
    print_out(f"log file path = {log_file_detailed}")
    print_out(f"date time now: {datetime_now()}")
    print_out("loading model...")
    start = time.time()

    ## Initialize agents
    if n_ray_actors > 1:
        AgentClass = (
            ray.remote(Agent).options(num_gpus=n_gpus_per_actor, num_cpus=n_gpus_per_actor * 3).remote
        )
    else:
        AgentClass = Agent

    ## ablation on partial cipher
    assert (
        args.partial_cipher_when_confident and args.partial_cipher_when_not_confident
    ) is False, "can not use them both!"
    if args.partial_cipher_when_confident:
        partial_cipher = "partial_cipher_when_confident"
    elif args.partial_cipher_when_not_confident:
        partial_cipher = "partial_cipher_when_not_confident"
    else:
        partial_cipher = ""

    if agents is None:
        agents = {}
        use_expert = any("expert" in agent_name for agent_name in debaters)

        for i, agent_name in enumerate(debaters):
            if other_agent_embedding:
                other_agent_name = debaters[1 - i]
            else:
                other_agent_name = None

            agent_name_clean = agent_name.replace("_dummy", "").replace("_expert", "")
            if agent_name_clean not in agents:
                agent_path = get_model_path(agent_name)

                agent = AgentClass(
                    initial_prompt_path=initial_prompt_paths[i],
                    debate_prompt_path=debate_prompt_paths[i],
                    engine=agent_name,
                    agent_path=agent_path,
                    dataset=dataset_name,
                    load_in_8bit=load_in_8bit,
                    max_new_tokens=max_new_tokens,
                    no_convert_ans_choice=no_convert_ans_choice,
                    debug=debug,
                    other_agent_embedding=other_agent_embedding,
                    other_agent_name=other_agent_name,
                    use_expert_or_dummy_expert=use_expert,
                    positional_bias=positional_bias,
                    partial_cipher=partial_cipher,
                    partial_thres=partial_thres,
                    use_entropy=partial_entropy_over_max,
                )

                agents[agent_name] = agent
                print_out(f"done loading {agent_path} in { (time.time()-start)/60} mins")

    temperatures_dict = {
        f"{debater}_{i}": temperatures[i * n_rounds : (i + 1) * n_rounds]
        for i, debater in enumerate(debaters)
    }
    print_out("--- temperatures_dict:" + str(temperatures_dict))

    generated_description = {}

    for i, batch in tqdm(enumerate(dataloader)):
        questions = batch["question"]
        gts = batch["answer"]

        questions = ensure_list(questions)
        gts = ensure_list(gts)

        print_out(f"================================================")
        print_out(f"batch {(i+1)}/{len(dataloader)}; total #questions={n_ques}")
        print_out("questions:")
        print_out(questions)
        print_out("ground truth: ")
        print_out(gts)

        if n_sols > 1:
            agent_name = debaters[0]
            agent = agents[agent_name]
            temp, top_p = temperatures[0], top_ps[0]
            res_dict = run_majority_vote(
                n_sols,
                questions=questions,
                gts=gts,
                agent_name=agent_name,
                agent=agent,
                temperature=temp,
                top_p=top_p,
                top_p_emb=top_p_emb,
                top_k_emb=top_k_emb,
                use_ray=n_ray_actors > 1,
            )
        else:
            res_dict = run_multiagent_debate(
                n_rounds=n_rounds,
                questions=questions,
                gts=gts,
                debaters=debaters,
                agents=agents,
                temperatures_dict=temperatures_dict,
                top_ps=top_ps,
                vector_language=vector_language,
                top_p_emb=top_p_emb,
                top_k_emb=top_k_emb,
                l2_norm=l2_norm,
                early_stop=early_stop,
                using_ray=n_ray_actors > 1,
                llama1_vs_llama2=other_agent_embedding,
            )
        generated_description.update(res_dict)

    new_dir = os_join(root_dir, "output", "log_" + datetime_short)
    if not os.path.exists(new_dir):
        # os.makedirs(new_dir)  ## permission denied error
        ## work around
        new_dir = os_join(root_dir, "output", "log_2023-others")

    # output_path = os_join(new_dir, output_name)
    # json.dump(generated_description, open(output_path, "w"))

    # print_out(f"wrote output json to {output_path}")
    # print_out(f"wrote full log to {log_file_detailed}")
    print_out(f"date time now: {datetime_now()}")
    end = time.time()
    print_out(f"total time: {round((end-start)/60,1)} mins")
    print_out("======Done!!!======")
    return generated_description


def exists(val):
    return val is not None


def default(val, default):
    return val if exists(val) else default


def debate(args: Optional[Namespace] = None, agents: Optional[Dict] = None) -> str:
    ## get arguments
    set_seeds(seed=args.seed)

    if args is None:
        args = get_args()
    print(args)

    n_ques = args.n_questions
    data_path = args.data_path
    batch_size = args.batch_size
    dataset_name = args.dataset
    n_ray_actors = args.n_ray_actors

    ## Prepare dataset
    if dataset_name == "gsm8k":
        dataset = GSM8K(input_path=data_path, n_ques=n_ques, seed=args.data_sample_seed)
    elif dataset_name == "mmlu":
        dataset = MMLU(input_path=data_path, n_ques=n_ques, seed=args.data_sample_seed)
    elif dataset_name == "arithmetic":
        dataset = Arithmetic(input_path=data_path, n_ques=n_ques, seed=args.data_sample_seed)
    elif dataset_name == "biography":
        dataset = Biography(input_path=data_path, n_ques=n_ques, seed=args.data_sample_seed)
    else:
        raise NotImplementedError()

    ## split the dataset into n_ray_actors
    dataset_list = []
    res = {}
    handlers = []

    if n_ray_actors > 1:
        debate_helper_func = ray.remote(debate_helper).remote
        lens = [len(dataset) // n_ray_actors] * (n_ray_actors - 1) + [
            len(dataset) - len(dataset) // n_ray_actors * (n_ray_actors - 1)
        ]
        dataset_list = random_split(dataset, lens)

        for sub_dataset in dataset_list:
            dataloader = DataLoader(
                sub_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False
            )
            handler = debate_helper_func(args, dataloader, agents)
            handlers.append(handler)

        while len(handlers):
            done_id, handlers = ray.wait(handlers)
            res.update(ray.get(done_id[0]))
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)
        generated_description = debate_helper(args, dataloader, agents)
        res.update(generated_description)

    datetime_short = datetime_now("%Y-%b-%d")
    new_dir = os_join(args.root_dir, "output", "log_" + datetime_short)
    if not os.path.exists(new_dir):
        try:
            os.makedirs(new_dir, exist_ok=True)  ## permission denied error
        except Exception as e:
            ## work around
            print(e)
            new_dir = os_join(args.root_dir, "output", "log_2023-others")

    # extract filename from `data_path`
    filename = data_path.split("/")[-1].split(".")[0]
    n_debaters = len(args.debaters)

    # temperatures = maybe_duplicate(args.temperatures, args.n_rounds * n_debaters)
    # top_ps = maybe_duplicate(args.top_ps, args.n_rounds)

    trial = os.environ.get("JOB_ID", "")
    temps = args.temperatures[:: args.n_rounds]
    temps_str = "-".join([str(round(float(x), 3)) for x in temps])
    # top_ps_str = "-".join([str(x) for x in top_ps])
    use_vector_language = "vec" if args.vector_language else "human"
    positional_bias_str = "_bias" if args.positional_bias else ""
    entropy = "_entropy" if args.partial_entropy_over_max else ""

    use_partial_cipher = ""
    if args.partial_cipher_when_not_confident:
        use_partial_cipher = "_partialNOT"
    elif args.partial_cipher_when_confident:
        use_partial_cipher = "_partial"

    partial_thres = args.partial_thres
    output_name = f"id{trial}_{n_debaters}{args.debaters[0]}_{args.debaters[1]}_{dataset_name}_{filename}_seed{args.seed}_{use_vector_language}_nsols{args.n_sols_each_ques}_r{args.n_rounds}_temp{temps_str}_nques{n_ques}_{datetime_now()}{positional_bias_str}{use_partial_cipher}{partial_thres}{entropy}.json"

    ## make sure it is fewer than 255 characters
    output_name = output_name[:255]

    output_path = os_join(new_dir, output_name)
    json.dump(res, open(output_path, "w"))

    print_out(f"wrote output json to {output_path}")

    return output_path


def read_as_list(file_path: str):
    data = []  # List to store the parsed data

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespaces
            if line:
                values = [float(x) for x in line.split(",")]
                data.append(values)

    return data


def get_search_space(config):
    n_debaters = len(config.debaters)

    if config.point_path is not None and config.point_path != "":
        ## read txt file, each line is a list of float numbers
        probe_points = read_as_list(config.point_path)
    else:
        probe_points = []
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))

    for i in range(config.num_points - len(probe_points)):
        ## generate `n_debaters` float numbers:
        ## temperature_1, temperature_2, ..., temperature_n_debaters
        temperatures = np.random.uniform(
            low=config.temperature_min, high=config.temperature_max, size=n_debaters
        )
        if len(config.debaters) == 2:
            ## sort the temperatures
            temperatures = sorted(temperatures)

        probe_points.append(temperatures)
    return probe_points


def evaluate_run(json_path: str, criterion: str) -> float:
    res_dict = agents_change_their_minds_analysis(json_path, n_rounds=None)
    acc = round(res_dict[criterion][1][0], 4)
    return acc


def run_baysian_opt(config):
    def run_debate_then_evaluate(**kawgs) -> float:
        ## hyper-params
        temperatures = [kawgs[f"temperature_{i+1}"] for i in range(len(config.debaters))]

        ## update temperature_1 in new args
        n_rounds = config.n_rounds
        config.temperatures = [-1] * n_rounds * len(config.debaters)
        for d in range(len(config.debaters)):
            config.temperatures[d * n_rounds : (d + 1) * n_rounds] = [temperatures[d]] * n_rounds

        ## check if new_config.temperatures is sorted accendingly
        if len(config.debaters) == 2 and sorted(config.temperatures) != config.temperatures:
            print("skip this run for 2 agents because order doesn't matter in this case")
            print(f"temps={config.temperatures}")
            acc = -1
        else:
            ## run debate

            json_path = debate(config)

            print("new_config")
            print(config)
            ## evaluate run
            acc = evaluate_run(json_path=json_path, criterion=criterion)
        return acc

    criterion = args.bayesian_criterion
    subdataname = config.data_path.split("/")[-1].split(".")[0]
    trial_id = os.environ.get("JOB_ID", "")
    is_vector = "_vector" if config.vector_language else "_human"
    file_name = f"{config.dataset}_{subdataname}_{is_vector}_{trial_id}_{len(config.debaters)}{config.debaters[0]}{config.debaters[1]}_seed{config.seed}"
    log_path = f"{config.root_dir}/output/bayesian_opt_v2/{file_name}"
    logger = JSONLogger(path=log_path)

    pbounds = {
        f"temperature_{i+1}": (config.temperature_min, config.temperature_max)
        for i in range(len(config.debaters))
    }
    optimizer = BayesianOptimization(
        f=run_debate_then_evaluate,
        pbounds=pbounds,
        random_state=config.seed,
        allow_duplicate_points=True,
    )

    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    if config.bayesian_json_log is not None and config.bayesian_json_log != "":
        print("Before loading checkpoint: optimizer is aware of {} points.".format(len(optimizer.space)))
        load_logs(optimizer, logs=[config.bayesian_json_log])
        print("Loaded checkpoint! The optimizer is now aware of {} points.".format(len(optimizer.space)))

    init_points = config.bayesian_num_init_points
    optimizer.maximize(init_points=init_points, n_iter=config.num_points - init_points)
    print("-----\nFinal result:", optimizer.max)


if __name__ == "__main__":
    ## set visible GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    with open("config.yml", "r") as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)

    args = get_args()
    original_seed = args.seed

    args.root_dir = yaml_config["root_dir"]
    if original_seed is None:
        args.seed = default(args.seed, random.randint(100, 100000))

    if args.n_ray_actors > 1:
        ray.init()
        print("+-+-+-+-+-+-+-+-+- ray resources: ", ray.available_resources())

    if args.num_points > 1:
        if args.use_bayesian:
            run_baysian_opt(args)
        else:  ## random search
            ## set seed for random, numpy
            if args.temperatures == []:
                temp_points_list = get_search_space(args)
            else:
                temp_points_list = [args.temperatures for _ in range(args.num_points)]
            print_out(f"-----++++++++++++ temp_points: {temp_points_list}")

            for i, temp_points in enumerate(temp_points_list):
                print_out(f"================== point {i+1}/{len(temp_points_list)} ==================")
                if len(temp_points) < len(args.debaters) * args.n_rounds:
                    args.temperatures = duplicate_temp(args.n_rounds, len(args.debaters), temp_points)
                else:
                    args.temperatures = temp_points
                if original_seed is None:
                    args.seed = random.randint(100, 100000)
                debate(args)
    else:
        if len(args.temperatures) < len(args.debaters) * args.n_rounds:
            args.temperatures = duplicate_temp(args.n_rounds, len(args.debaters), args.temperatures)
        debate(args)
