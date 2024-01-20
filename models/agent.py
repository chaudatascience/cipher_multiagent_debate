from transformers import (
    AutoTokenizer,  # type: ignore
    LlamaForCausalLM,
    LlamaTokenizer,  # type: ignore
    FalconForCausalLM,  # type: ignore
    MptForCausalLM,  # type: ignore
)

from optimum.bettertransformer import BetterTransformer

## transformer 4.30.1, where only LLaMA is supported (LLaMA2 hasn't been released yet)
# from models.llama_hf import LlamaHFv1

from copy import deepcopy
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, List, Optional, Dict, Union
import yaml
from einops import rearrange
import json
import re
import numpy as np

pprint = lambda x: print(json.dumps(x, indent=2, ensure_ascii=False))


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample, device):
    def _move_to_cuda(tensor):
        return tensor.to(device)

    return apply_to_sample(_move_to_cuda, sample)


class Agent:
    def __init__(
        self,
        initial_prompt_path: str,
        debate_prompt_path: str,
        engine: str,
        agent_path: str,
        dataset: str,
        no_convert_ans_choice: bool,
        load_in_8bit: bool = False,
        max_new_tokens: int = 500,
        debug: bool = False,
        other_agent_embedding: bool = False,
        other_agent_name: Union[str, None] = None,
        use_expert_or_dummy_expert: bool = False,
        positional_bias: bool = False,
        partial_cipher: str = "",
        partial_thres: Union[float, None] = None,
        use_entropy: bool = False,
    ) -> None:
        self.engine = engine.lower()
        self.agent_path = agent_path
        self.max_new_tokens = max_new_tokens
        self.load_in_8bit = load_in_8bit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.no_convert_ans_choice = no_convert_ans_choice
        self.debug = debug
        self.use_expert_or_dummy_expert = use_expert_or_dummy_expert
        self.partial_cipher = partial_cipher
        self.partial_thres = partial_thres
        self.use_entropy = use_entropy

        ## PLACEHOLDER
        self.placeholders = {}
        self.placeholders["question"] = "{QUESTION_PLACEHOLDER}"
        self.placeholders["other_sol"] = "{OTHER_SOLUTION_PLACEHOLDER}"
        self.placeholders["my_sol"] = "{MY_SOLUTION_PLACEHOLDER}"

        ## Separate contents and prompt
        sep_token_ids = {"llama": {}}
        sep_token_ids["llama"]["backtick_start"] = 7521  ## : ```...
        sep_token_ids["llama"]["backtick_end"] = 28956  ## ...```\n
        sep_token_ids["llama"]["answer"] = 22550
        sep_token_ids["llama"]["double_enters"] = torch.tensor([13, 13], device=self.device)
        sep_token_ids["llama"]["your_solution"] = torch.tensor([10858, 1650, 29901], device=self.device)
        sep_token_ids["llama"]["correct_solution"] = torch.tensor(
            [12521, 1621, 1650, 29901], device=self.device
        )
        sep_token_ids["llama"]["lets_think_step_by_step"] = torch.tensor(
            [12024, 29915, 29879, 1348, 4331, 491, 4331, 29901, 13], device=self.device
        )

        sep_token_ids.update({"falcon": {}})
        sep_token_ids["falcon"]["backtick_start"] = 17593  ## : ```...
        sep_token_ids["falcon"]["backtick_end"] = 17593  ## ...```\n
        sep_token_ids["falcon"]["answer"] = 20309
        sep_token_ids["falcon"]["double_enters"] = torch.tensor([193, 193], device=self.device)
        sep_token_ids["falcon"]["your_solution"] = torch.tensor([4560, 3377, 37, 204], device=self.device)
        sep_token_ids["falcon"]["correct_solution"] = torch.tensor([42545, 3377, 37, 204], device=self.device)
        sep_token_ids["falcon"]["lets_think_step_by_step"] = torch.tensor(
            [5400, 18, 94, 864, 2006, 431, 2006, 37, 193], device=self.device
        )

        sep_token_ids.update({"mpt": {}})
        sep_token_ids["mpt"]["backtick_start"] = 5190  ## : ```...
        sep_token_ids["mpt"]["backtick_end"] = 11202  ## ...```\n
        sep_token_ids["mpt"]["answer"] = 32869
        sep_token_ids["mpt"]["double_enters"] = torch.tensor([187, 187], device=self.device)
        sep_token_ids["mpt"]["your_solution"] = torch.tensor([7093, 2900, 27, 2634], device=self.device)
        sep_token_ids["mpt"]["correct_solution"] = torch.tensor([47390, 2900, 27, 2634], device=self.device)
        sep_token_ids["mpt"]["lets_think_step_by_step"] = torch.tensor(
            [1466, 434, 1158, 3213, 407, 3213, 27, 209, 187], device=self.device
        )

        sep_token_ids.update({"wizardmath": {}})
        sep_token_ids["wizardmath"]["backtick_start"] = 7521
        sep_token_ids["wizardmath"]["backtick_end"] = 2
        ## technically, this is </s>, not backtick, but we don't use backtick (we follow wizardmath prompt template)

        sep_token_ids["wizardmath"]["answer"] = 1234
        sep_token_ids["wizardmath"]["double_enters"] = sep_token_ids["llama"]["double_enters"]  # "\n\n"
        sep_token_ids["wizardmath"]["your_solution"] = torch.tensor(
            [2277, 29937, 3575, 1650, 29901], device=self.device
        )  # "### Your solution: "
        sep_token_ids["wizardmath"]["correct_solution"] = torch.tensor(
            [2277, 29937, 13291, 29901], device=self.device
        )  # "### Response: "
        sep_token_ids["wizardmath"]["lets_think_step_by_step"] = torch.tensor(
            [2803, 29915, 29879, 1348, 4331, 491, 4331, 29889], device=self.device
        )  # "Let's think step by step."

        sep_token_ids["wizardmath"]["instruct_wizardmath"] = torch.tensor(
            [
                13866,
                338,
                385,
                15278,
                393,
                16612,
                263,
                3414,
                29889,
                14350,
                263,
                2933,
                393,
                7128,
                2486,
                1614,
                2167,
                278,
                2009,
                29889,
                13,
                13,
            ],
            device=self.device,
        )  # "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"

        if "llama" in self.engine:
            self.sep_token_ids = sep_token_ids["llama"]
        elif "falcon" in self.engine:
            self.sep_token_ids = sep_token_ids["falcon"]
        elif "mpt" in self.engine:
            self.sep_token_ids = sep_token_ids["mpt"]
        elif "wizardmath" in self.engine:
            self.sep_token_ids = sep_token_ids["wizardmath"]
        else:
            self.sep_token_ids = None

        ## set up prompt
        self.initial_prompt = self._setup_prompt_from_examples_file(initial_prompt_path)
        self.debate_prompt = self._setup_prompt_from_examples_file(debate_prompt_path)

        self.positional_bias = positional_bias

        if "llama-2" in self.engine or "wizardmath" in self.engine:
            self.context_length = 4096
        elif "mpt" in self.engine:
            self.context_length = 8192
        else:
            self.context_length = 2048
        self.other_emb = None
        if "llama" in self.engine or "wizardmath" in self.engine:
            # ## Hugging face changed the code of LLaMA class to support LLaMA2.
            # ## to make sure the results are reproducible on LLaMA1, I keep the old version of LLaMA class (i.e, LlamaHFv1) as a separate file.
            # if "llama-2" in self.engine:
            #     LLaMA = LlamaForCausalLM  ## support LLaMA2. transformer 4.31.0
            # else:
            #     LLaMA = LlamaHFv1  ## only LLaMA. transformer 4.30.0

            print("self.context_length", self.context_length)
            LLaMA = LlamaForCausalLM
            self.tokenizer = LlamaTokenizer.from_pretrained(agent_path)
            self.tokenizer.pad_id = self.tokenizer.eos_token_id

            # https://github.com/huggingface/transformers/blob/476be08c4aa96f8c1cae4200d2677bbe8f12cf80/src/transformers/models/llama/modeling_llama.py#L727C1-L727C1
            self.agent = LLaMA.from_pretrained(
                agent_path,
                load_in_8bit=self.load_in_8bit,
                device_map="auto",
                torch_dtype=torch.float16,
            )  ##.to("cuda")
            ## emb table
            self.emb_table = self.agent.model.embed_tokens  ## type: ignore [32000, d]

            if self.dataset == "mmlu":
                abcd_ids = self.tokenizer(["A", "B", "C", "D"], add_special_tokens=False)["input_ids"]
                abcd_ids = [item for sublist in abcd_ids for item in sublist]
                self.emb_choices = F.normalize(
                    self.emb_table(torch.tensor(abcd_ids, device=self.device)), p=2, dim=-1
                ).float()
                self.choices_mapper = {"0": "A", "1": "B", "2": "C", "3": "D"}
            self.emb_table_norm = F.normalize(self.emb_table.weight, p=2, dim=-1)  ## type: ignore

            with open("config.yml", "r") as yaml_file:
                yaml_config = yaml.safe_load(yaml_file)

            if other_agent_embedding:
                try:
                    other_emb_path = yaml_config["other_emb_path"].format(
                        other_agent_name=other_agent_name.lower()
                    )
                    self.other_emb = torch.load(other_emb_path).to(self.device)
                except:
                    other_emb_path = f"./emb_weights/{other_agent_name.lower()}.pt"
                    self.other_emb = torch.load(other_emb_path).to(self.device)

                # self.emb_table.weight.requires_grad = False
                # torch.save(self.emb_table.weight, other_emb_path)
        elif "falcon" in self.engine:
            self.tokenizer = AutoTokenizer.from_pretrained(agent_path)
            agent = FalconForCausalLM.from_pretrained(
                agent_path,
                # torch_dtype=torch.bfloat16,
                torch_dtype=torch.float16,
                device_map="balanced",
                load_in_8bit=self.load_in_8bit,
            )  ## .to("cuda")

            self.agent = BetterTransformer.transform(agent, keep_original_model=True)

            # Setting `pad_token_id` to `eos_token_id`:11 for open-end generation.
            self.tokenizer.pad_id = self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.emb_table = self.agent.transformer.word_embeddings
            self.emb_table_norm = F.normalize(self.emb_table.weight, p=2, dim=-1)

        elif "mpt" in self.engine:
            self.tokenizer = AutoTokenizer.from_pretrained(agent_path)
            self.agent = MptForCausalLM.from_pretrained(
                agent_path,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=self.load_in_8bit,
            )
            self.tokenizer.pad_id = self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.emb_table = self.agent.transformer.wte  # Embedding(50432, 7168)
            self.emb_table_norm = F.normalize(self.emb_table.weight, p=2, dim=-1)
        else:
            raise ValueError(f"Unknown engine {engine}")
        print(f"Loaded model from {agent_path}")

    def _setup_prompt_from_examples_file(self, prompt_path) -> str:
        with open(prompt_path, "r") as f:
            template_prompt = f.read()
        return template_prompt

    def make_initial_prompt(self, question: str) -> str:
        question = question.strip()
        prompt = self.initial_prompt.replace(self.placeholders["question"], question)
        return prompt

    def make_debate_prompt(
        self, question: str, prev_sols: Tuple[str], agent_idx: int, vector_language: bool
    ) -> str:
        question = question.strip()
        prompt = self.debate_prompt.replace(self.placeholders["question"], question)

        if not vector_language:
            prev_sols = list(prev_sols)  ## type: ignore
            my_sol = prev_sols[agent_idx]
            other_sols = prev_sols[:agent_idx] + prev_sols[agent_idx + 1 :]

            for i, other_sol in enumerate(other_sols):
                placeholder_i = self.placeholders["other_sol"].replace("}", f"_{i}" + "}")
                prompt = prompt.replace(placeholder_i, other_sol)

            prompt = prompt.replace(self.placeholders["my_sol"], my_sol)

        return prompt

    def _generate_single_output(self, prompt: str, temperature: float, top_p: float) -> str:
        """
        Generate output for a single question, using Hugging Face functions
        """

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.context_length
        ).to(self.device)

        print(f"num tokens: {len(inputs.input_ids[0])}, temperature: {temperature}, top_p: {top_p}")

        generate_ids = self.agent.generate(  ## type: ignore
            inputs.input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]

        ## only show newly generated tokens
        prompt_len = len(prompt)
        res = output[prompt_len:]

        return res

    def _expert_solution(self, vector_language: bool, gt_for_expert: List[str], prompts: List[str]):
        print("expert answer")
        if vector_language:
            gt_clean = [re.sub(r"<<.*?>>", "", gt).replace("\n####", "\nAnswer:") for gt in gt_for_expert]
            return {
                "emb": self._text_to_emb(gt_clean, False),
                "nearest_neighbor_texts": gt_clean,
                "human_readable_texts": "",
                "prompt": prompts,
            }
        else:
            return gt_for_expert, prompts

    def give_first_solutions(
        self,
        questions: List[str],
        temperature: float,
        top_p: float,
        vector_language: bool,
        top_p_emb: float,
        top_k_emb: int,
        l2_norm: float,
        early_stop: bool,
        gt_for_expert: Optional[List[str]] = None,
        convert_to_cpu: bool = False,
    ) -> Union[Dict, Tuple]:
        prompts = [self.make_initial_prompt(ques) for ques in questions]

        if gt_for_expert is not None:
            assert self.use_expert_or_dummy_expert, "Only provide gt for expert model!"

        if self.use_expert_or_dummy_expert:
            assert gt_for_expert is not None, "need to provide gt for expert model!"
            return self._expert_solution(
                vector_language=vector_language, gt_for_expert=gt_for_expert, prompts=prompts
            )

        elif (
            self.engine.startswith("llama")
            or self.engine.startswith("falcon")
            or self.engine.startswith("mpt")
            or self.engine.startswith("wizardmath")
        ):
            if vector_language:
                return self.generate_output_embs(
                    prompts=prompts,
                    temperature=temperature,
                    is_debate=False,
                    other_sol_embs_list=None,
                    my_sol_embs=None,
                    top_p_emb=top_p_emb,
                    l2_norm=l2_norm,
                    top_k_emb=top_k_emb,
                    early_stop=early_stop,
                    convert_to_cpu=convert_to_cpu,
                )

            else:
                generated_text, generated_text_ans_token = self.generate_outputs_human(
                    prompts, temperature=temperature, top_p=top_p, early_stop=early_stop
                )

                return generated_text, generated_text_ans_token, prompts
        else:
            decoded_strs = [
                self._generate_single_output(prompt, temperature=temperature, top_p=top_p)
                for prompt in prompts
            ]
            generated_text = generated_text_ans_token = decoded_strs
            return generated_text, generated_text_ans_token, prompts

    def give_debate_solutions(
        self,
        questions: List[str],
        prev_sols_batch: List,
        other_sols_batch: List,
        agent_index: int,
        temperature: float,
        top_p: float,
        vector_language: bool,
        top_p_emb: float,
        top_k_emb: int,
        l2_norm: float,
        early_stop: bool,
        gt_for_expert: Optional[List[str]] = None,
        convert_to_cpu: bool = False,
    ):
        """
        questions: `batch` questions
        prev_sols_batch: `batch` entries, each entry is a tuple of `num_agents` solutions
        """
        ## more prev_sols_batch to the same device
        if convert_to_cpu and vector_language:  ## prev_sols_batch may in CPU now, needed to move to GPU
            ## move prev_sol_batch to cuda
            prev_sols_batch_cuda = []
            for i in range(len(prev_sols_batch)):
                tmp = []
                for j in range(len(prev_sols_batch[i])):
                    tmp.append(prev_sols_batch[i][j].to(self.device))
                prev_sols_batch_cuda.append(tuple(tmp))
            prev_sols_batch = prev_sols_batch_cuda

            if other_sols_batch is not None:
                other_sol_history_batch_cuda = []
                for i in range(len(other_sols_batch)):
                    tmp = []
                    for j in range(len(other_sols_batch[i])):
                        tmp.append(other_sols_batch[i][j].to(self.device))
                    other_sol_history_batch_cuda.append(tuple(tmp))
                other_sols_batch = other_sol_history_batch_cuda

        prompts = [
            self.make_debate_prompt(
                ques, prev_sols=prev_sols, agent_idx=agent_index, vector_language=vector_language
            )
            for ques, prev_sols in zip(questions, prev_sols_batch)
        ]

        if gt_for_expert is not None:
            assert self.use_expert_or_dummy_expert, "Only provide gt for expert/dummy_expert model!"
        if self.use_expert_or_dummy_expert:
            assert gt_for_expert is not None, "need to provide gt for expert model!"
            return self._expert_solution(
                vector_language=vector_language, gt_for_expert=gt_for_expert, prompts=prompts
            )

        elif (
            self.engine.startswith("llama")
            or self.engine.startswith("falcon")
            or self.engine.startswith("mpt")
            or self.engine.startswith("wizardmath")
        ):
            if vector_language:
                my_sol_emb_batch = [sol[agent_index] for sol in prev_sols_batch]  # each is a tensor
                if other_sols_batch is None:
                    other_sols_emb_batch = [
                        list(sol[:agent_index] + sol[agent_index + 1 :]) for sol in prev_sols_batch
                    ]  ## each element is a list of tensor (i.e., `num_agents` - 1 tensors)
                else:
                    other_sols_emb_batch = [
                        list(sol[:agent_index] + sol[agent_index + 1 :]) for sol in other_sols_batch
                    ]  ## each element is a list of tensor (i.e., `num_agents` - 1 tensors)
                    ## note, this work for 2 agents debate where one is llama 1 and the other is llama2.
                    ## for more than 3 agents debate that are different llama versions, we may need to change this.

                return self.generate_output_embs(
                    prompts=prompts,
                    temperature=temperature,
                    is_debate=True,
                    my_sol_embs=my_sol_emb_batch,
                    other_sol_embs_list=other_sols_emb_batch,
                    top_p_emb=top_p_emb,
                    l2_norm=l2_norm,
                    top_k_emb=top_k_emb,
                    early_stop=early_stop,
                    convert_to_cpu=convert_to_cpu,
                )
            else:
                generated_text, generated_text_ans_token = self.generate_outputs_human(
                    prompts, temperature=temperature, top_p=top_p, early_stop=early_stop
                )
                return generated_text, generated_text_ans_token, prompts
        else:
            decoded_strs = [
                self._generate_single_output(prompt, temperature=temperature, top_p=top_p)
                for prompt in prompts
            ]
            generated_text = generated_text_ans_token = decoded_strs
            return generated_text, generated_text_ans_token, prompts

    def _prompts_to_tokens(self, prompts: List[str], is_bos: bool) -> List[Tensor]:
        prompt_tokens = [
            (
                self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.context_length,
                )
            ).input_ids.reshape(-1)
            for prompt in prompts
        ]  ## list of b elements, each element is a tensor of size (t,), where t varies

        if not is_bos:
            prompt_tokens = [t[1:] for t in prompt_tokens]
        return prompt_tokens

    def _text_to_emb(self, prompts: List[str], is_bos: bool) -> List[Tensor]:
        prompt_tokens = self._prompts_to_tokens(prompts, is_bos=is_bos)
        token_embs = [self.emb_table(token) for token in prompt_tokens]
        return token_embs

    def _generate_tokens(self, prompt_tokens: List[Tensor], temperature: float, top_p: float) -> Tensor:
        bsz = len(prompt_tokens)

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = self.max_new_tokens + max_prompt_size
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()  ## type: ignore
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = t
        input_text_mask = tokens != self.tokenizer.pad_id  ## type: ignore
        start_pos = min_prompt_size
        prev_pos = 0
        past_key_values = None

        for cur_pos in range(start_pos, total_len):
            with torch.inference_mode():
                if self.engine.startswith("llama"):
                    position_ids = torch.arange(prev_pos, cur_pos).long()
                    outputs = self.agent.forward(  # type: ignore
                        input_ids=tokens[:, prev_pos:cur_pos],  ## type: ignore
                        position_ids=position_ids,  ## type: ignore
                        past_key_values=past_key_values,
                    )  ## (b, 32000)

                elif self.engine.startswith("falcon"):
                    outputs = self.agent.forward(  # type: ignore
                        input_ids=tokens[:, prev_pos:cur_pos],  ## type: ignore
                        past_key_values=past_key_values,
                    )

                elif self.engine.startswith("mpt"):
                    outputs = self.agent.forward(  # type: ignore
                        input_ids=tokens[:, prev_pos:cur_pos],  ## type: ignore
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                elif self.engine.startswith("wizardmath"):
                    position_ids = torch.arange(prev_pos, cur_pos).long()
                    outputs = self.agent.forward(  # type: ignore
                        input_ids=tokens[:, prev_pos:cur_pos],  ## type: ignore
                        position_ids=position_ids,  ## type: ignore
                        past_key_values=past_key_values,
                        use_cache=True,
                    )  ## (b, 32000)

                logits, past_key_values = (
                    outputs.logits[:, -1, :],  ## type: ignore
                    outputs.past_key_values,  ## type: ignore
                )  ## type: ignore
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)  ## (b, 32000)
                else:
                    probs = torch.softmax(logits, dim=-1)  ## (b, 32000)

                if self.debug:
                    probs_no_temp = torch.softmax(logits, dim=-1)
                    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                    probs_no_temp_sort, probs_no_temp_idx = torch.sort(probs_no_temp, dim=-1, descending=True)
                    k_tokens = 10
                    top_k_tokens_idx = probs_idx[:, :k_tokens]
                    top_k_tokens = [self.tokenizer.decode(idx) for idx in top_k_tokens_idx]

            if temperature > 0:
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            if self.debug:
                print(
                    top_k_tokens,  ## type: ignore
                    "->\t",
                    self.tokenizer.batch_decode(next_token, skip_special_tokens=True),
                    "probs_temp",
                    np.round(probs_sort.cpu().numpy()[:k_tokens], 2),  ## type: ignore
                    "\t",
                    "prob_no_temp",
                    np.round(probs_no_temp_sort.cpu().numpy()[:k_tokens], 2),  ## type: ignore
                )

            next_token = next_token.reshape(-1)  ## [b,1] -> [b]
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token  ## [b, 265]

            prev_pos = cur_pos
        return tokens

    def _decode_tokens(
        self,
        tokens: Union[Tensor, List[Tensor]],
        prompt_lens: List[int],
        max_new_tokens: int,
        early_stop: bool,
    ) -> Tuple[List[str], List[int]]:
        decoded = []
        stop_idxs_list = []
        ## cut off the answer when it meets one of these tokens

        backtick_end = self.sep_token_ids["backtick_end"]
        eos = self.tokenizer.eos_token_id

        ## replace all 32000 in tokens by 2
        if self.engine.startswith("wizardmath"):
            tokens = torch.where(tokens == 32000, eos, tokens)

        for i, t in enumerate(tokens):
            # cut from new generated token to max gen len
            t = t[prompt_lens[i] : prompt_lens[i] + max_new_tokens]

            # cut to eos/backtick toks if any
            early_stop_mask = t == backtick_end
            eos_mask = t == eos
            ans_token_mask = t == self.sep_token_ids["answer"]

            ans_token_idxs = torch.where(ans_token_mask)[0]
            eos_idxs = torch.where(eos_mask)[0]

            mask = early_stop_mask | eos_mask
            stop_idxs = torch.where(mask)[0]

            stop_index = len(t)
            if early_stop:
                if len(stop_idxs) > 0:
                    ## if "Answer: something" in the answer,
                    ## we'll chop off at the first stop mask (i.e., backtick and eos, if any) after it.
                    if len(ans_token_idxs) > 0:
                        ans_token_index = ans_token_idxs[0]  ## pick the first index it appears
                        idx_of_stop_idxs = torch.where(ans_token_index < stop_index)[0]
                        if len(idx_of_stop_idxs) > 0:
                            idx_of_stop_index = idx_of_stop_idxs[0]
                            stop_index = stop_idxs[idx_of_stop_index]
                    elif (
                        self.engine.startswith("wizardmath") and len(eos_idxs) > 0
                    ):  ## cut off for wizardmath
                        stop_index = eos_idxs[0]
            else:
                if len(eos_idxs) > 0:
                    stop_index = eos_idxs[0]

            t = t[:stop_index]
            decoded.append(self.tokenizer.decode(t))
            stop_idxs_list.append(stop_index)
        return decoded, stop_idxs_list  ## decoded texts, stop index after removing prompts

    def generate_outputs_human(
        self, prompts: List[str], temperature: float, top_p: float, early_stop: bool
    ) -> Tuple[List[str], List[str]]:
        prompt_tokens = self._prompts_to_tokens(prompts, is_bos=True)
        prompt_lens = [len(t) for t in prompt_tokens]
        all_generated_tokens = self._generate_tokens(prompt_tokens, temperature, top_p)
        generated_text, stop_idxs = self._decode_tokens(
            all_generated_tokens, prompt_lens, self.max_new_tokens, early_stop=early_stop
        )

        generated_tokens = self.remove_promp_emb(all_generated_tokens, prompt_lens)

        generated_token_embs = [emb[:stop] for emb, stop in zip(generated_tokens, stop_idxs)]
        if not self.no_convert_ans_choice:
            generated_text_token_choice = self._nearest_token_choice(
                generated_text, generated_token_embs, vector=False
            )
        else:
            generated_text_token_choice = generated_text

        return generated_text, generated_text_token_choice

    @torch.inference_mode()
    def _generate_embs(
        self,
        prompt_embs: List[Tensor],
        temperature: float,
        top_p_emb: float,
        l2_norm: float,
        top_k_emb: int,
    ) -> Tuple:
        bsz, dim = len(prompt_embs), prompt_embs[0].shape[-1]
        prompt_lens = [len(t) for t in prompt_embs]
        total_len = self.max_new_tokens + max(prompt_lens)

        all_token_embs = torch.full(
            (bsz, total_len, dim), self.tokenizer.pad_id, device=self.device  ## type: ignore
        ).to(torch.float16)
        input_text_mask = torch.full((bsz, total_len), False, device=self.device).to(bool)  # type: ignore

        for i, len_token in enumerate(prompt_lens):
            all_token_embs[i, :len_token] = prompt_embs[i]
            input_text_mask[i, :len_token] = True

        token_embs = all_token_embs[:, : min(prompt_lens)]
        if self.other_emb is not None:
            token_embs_llama1_vs_llama2 = deepcopy(all_token_embs)
        else:
            token_embs_llama1_vs_llama2 = None

        start_pos = min(prompt_lens)
        prev_pos = 0
        past_key_values = None

        for cur_pos in range(start_pos, total_len):
            with torch.inference_mode():
                if self.engine.startswith("llama"):
                    position_ids = torch.arange(prev_pos, cur_pos).long()
                    outputs = self.agent.forward(  # type: ignore
                        input_ids=None,  ## type: ignore
                        inputs_embeds=token_embs,  ## type: ignore
                        position_ids=position_ids,  ## type: ignore
                        past_key_values=past_key_values,
                    )  ## (b, 32000)
                elif self.engine.startswith("falcon"):
                    outputs = self.agent.forward(  # type: ignore
                        input_ids=None,  ## type: ignore
                        inputs_embeds=token_embs,  ## type: ignore
                        past_key_values=past_key_values,
                    )
                elif self.engine.startswith("mpt"):
                    outputs = self.agent.forward(  # type: ignore
                        input_ids=None,  ## type: ignore
                        inputs_embeds=token_embs,  ## type: ignore
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                elif self.engine.startswith("wizardmath"):
                    position_ids = torch.arange(prev_pos, cur_pos).long()
                    outputs = self.agent.forward(  # type: ignore
                        input_ids=None,  ## type: ignore
                        inputs_embeds=token_embs,  ## type: ignore
                        position_ids=position_ids,  ## type: ignore
                        past_key_values=past_key_values,
                        use_cache=True,
                    )  ## (b, 32000)
                else:
                    raise ValueError(f"Haven't supported for engine {self.engine}")
                logits, past_key_values = (
                    outputs.logits[:, -1, :],  ## type: ignore
                    outputs.past_key_values,  ## type: ignore
                )
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)  ## (b, 32000)
                else:
                    probs = torch.softmax(logits, dim=-1)  ## (b, 32000)

            if self.debug:
                probs_no_temp = torch.softmax(logits, dim=-1)
                probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                probs_no_temp_sort, probs_no_temp_idx = torch.sort(probs_no_temp, dim=-1, descending=True)
                k_tokens = 10
                top_k_tokens_idx = probs_idx[:, :k_tokens]
                top_k_tokens = [self.tokenizer.decode(idx) for idx in top_k_tokens_idx]

            if top_p_emb < 1.0 or top_k_emb != -1:
                probs = self._get_probs_after_top_p_k_emb(probs, top_p_emb, top_k_emb)

            if self.partial_cipher == "":
                next_token_emb = torch.einsum(
                    "b v , v d -> b d", probs.to(torch.float16), self.emb_table.weight
                )  ## (b, 4096);  v: vocab_size, d: hid_dim
            else:  ## partial cipher for ablation
                probs_no_temp = torch.softmax(logits, dim=-1)  # shape (b, v)
                if self.use_entropy:
                    criteria = Categorical(probs=probs_no_temp).entropy().reshape(-1, 1)
                else:
                    criteria = torch.amax(probs_no_temp, dim=-1, keepdim=True)  # shape (b, 1)

                mask_data_binary = (
                    probs >= criteria
                )  ## # shape (b, v), one-hot vectors to keep the most confident token

                assert self.partial_cipher in [
                    "partial_cipher_when_not_confident",
                    "partial_cipher_when_confident",
                ], "partial_cipher must be one of the two options"

                if self.partial_cipher == "partial_cipher_when_not_confident":
                    mask_batch = (
                        criteria >= self.partial_thres
                    )  ## shape (b, 1), mark rows in batch that the model is very confident
                else:
                    mask_batch = criteria < self.partial_thres

                mask_probs = ~mask_batch  # shape (b, 1)

                probs = mask_probs * probs + mask_batch * mask_data_binary
                next_token_emb = torch.einsum(
                    "b v , v d -> b d", probs.to(torch.float16), self.emb_table.weight
                )  ## (b, 4096);  v: vocab_size, d: hid_dim

            if self.other_emb is not None:
                next_token_emb_llama1_vs_llama2 = torch.einsum(
                    "b v , v d -> b d", probs.to(torch.float16), self.other_emb
                )

            if self.debug:
                entropy = Categorical(probs=probs_no_temp).entropy()

                print(
                    top_k_tokens,  ## type: ignore
                    "->\t",
                    top_k_tokens[0][0],  ## type: ignore
                    "\t",
                    "probs_temp",
                    np.round(probs_sort.cpu().numpy()[:k_tokens], 2),  ## type: ignore
                    "\t",
                    "prob_no_temp",
                    np.round(probs_no_temp_sort.cpu().numpy()[:k_tokens], 2),
                    f"entropy={round(entropy.item(),3)}",  ## type: ignore
                )

            if l2_norm:
                next_token_emb = F.normalize(next_token_emb, p=2, dim=-1)

            for i in range(bsz):
                if not input_text_mask[i, cur_pos]:
                    all_token_embs[i, cur_pos] = next_token_emb[i]
                    if self.other_emb is not None:
                        token_embs_llama1_vs_llama2[i, cur_pos] = next_token_emb_llama1_vs_llama2[i]  # type: ignore

            token_embs = all_token_embs[:, cur_pos : cur_pos + 1, :]  ## (b, 1, 4096)
            prev_pos = cur_pos

        human_readable_text = ""  ## skip for now, no use
        return all_token_embs, human_readable_text, token_embs_llama1_vs_llama2

    @torch.inference_mode()
    def _decode_embs_nearest_neighbour(
        self, token_embs: Tensor, prompt_lens: List[int], max_new_tokens: int, early_stop: bool
    ) -> Tuple[List[str], List[int]]:
        """
        Find the nearest neighbour
        """
        ## l2 norm
        token_embs = F.normalize(token_embs, p=2, dim=-1)

        ## compute distance from t generated embeddings to `emb_table``
        dist = torch.cdist(token_embs, self.emb_table_norm, p=2)  ## [t, 32000]

        ## get the index of the most similarity
        next_tokens = torch.argmin(dist, dim=-1)  ## [t]

        decoded, stop_idxs = self._decode_tokens(next_tokens, prompt_lens, max_new_tokens, early_stop)
        return decoded, stop_idxs

    def _nearest_token_choice(
        self, nearest_neighbor_texts: List[str], generated_token_embs: List[Tensor], vector: bool
    ) -> List[str]:
        if self.dataset == "mmlu":
            try:
                last_embs_list = []
                for emb in generated_token_embs:
                    if len(emb) > 0:
                        if vector:
                            last_embs_list.append(emb[-1])
                        else:
                            last_embs_list.append(self.emb_table(emb[-1]))
                    else:
                        print("-----Warning: empty emb -----")
                        emb_random = torch.rand_like(self.emb_table(torch.tensor(0)), device=self.device)
                        last_embs_list.append(emb_random)

                last_embs = torch.stack(last_embs_list, dim=0)

                ## norm last_embs
                last_embs = F.normalize(last_embs, p=2, dim=-1).float()

                dist = torch.cdist(last_embs, self.emb_choices, p=2)
                min_dists, min_idxs = torch.min(dist, dim=1)
                nearest_option_choice = [self.choices_mapper[str(idx.item())] for idx in min_idxs]
                ## modify each element in nearest_neighbor_texts by replace the last letter by nearest_option_choice
                updated_nearest_neighbor_texts = []
                for text, choice in zip(nearest_neighbor_texts, nearest_option_choice):
                    updated_text = text[:-1] + choice
                    updated_nearest_neighbor_texts.append(updated_text)

                return updated_nearest_neighbor_texts
            except Exception as e:
                print(f"Warning: Exception {e}")
                return nearest_neighbor_texts
        else:
            ## do nothing
            return nearest_neighbor_texts

    @torch.inference_mode()
    def _get_debate_prompt_embs(
        self,
        prompts: List[str],
        other_sol_embs_list: List[List[Tensor]],
        my_sol_embs: List[Tensor],
    ) -> List[Tensor]:
        def get_agent_k_solution_template(agent_idx: int):
            agent_k = self.tokenizer(f"{agent_idx}").input_ids[-1]
            if self.engine.startswith("llama"):
                agent_ks_solution = torch.tensor(
                    [19661, 29871, agent_k, 29915, 29879, 1650, 29901], device=self.device
                )
            elif self.engine.startswith("falcon"):
                agent_ks_solution = torch.tensor(
                    [28796, 204, agent_k, 18, 94, 3377, 37, 204], device=self.device
                )
            elif self.engine.startswith("mpt"):
                agent_k = self.tokenizer(f" {agent_idx}'s").input_ids[0]
                agent_ks_solution = torch.tensor([28172, agent_k, 434, 2900, 27, 2634], device=self.device)
            elif self.engine.startswith("wizardmath"):
                # agent_k = self.tokenizer(f" {agent_idx}").input_ids[0]
                agent_ks_solution = torch.tensor(
                    [2277, 29937, 28330, 29871, agent_k, 29915, 29879, 1650, 29901],
                    device=self.device,
                )
            else:
                raise ValueError(f"Unsupported engine {self.engine} yet")
            agent_ks_solution_emb = self.emb_table(agent_ks_solution)
            return agent_ks_solution_emb

        def get_debate_promp_emb(debate_prompt: str, _other_sol_embs: List[Tensor], _my_sol_emb: Tensor):
            if self.engine.startswith("wizardmath"):
                debate_prompt_emb = self._text_to_emb([debate_prompt], is_bos=False)[0]
            else:
                debate_prompt_emb = self._text_to_emb([debate_prompt], is_bos=True)[0]
            all_other_sols = []
            if not self.engine.startswith("wizardmath"):
                for agent_idx, _other_sol_emb in enumerate(_other_sol_embs, 1):
                    agent_i_content = [
                        get_agent_k_solution_template(agent_idx),
                        backtick_start_emb,
                        _other_sol_emb,
                        backtick_end_emb,
                        double_enters_emb,
                    ]
                    all_other_sols.extend(agent_i_content)
            else:  ## wizardmath uses different prompts
                for agent_idx, _other_sol_emb in enumerate(_other_sol_embs, 1):
                    agent_i_content = [
                        get_agent_k_solution_template(agent_idx),
                        _other_sol_emb,
                        double_enters_emb,
                    ]
                    all_other_sols.extend(agent_i_content)

            if not self.engine.startswith("wizardmath"):
                if self.positional_bias:
                    res = torch.concat(
                        [
                            debate_prompt_emb,
                            your_solution_emb,
                            backtick_start_emb,
                            _my_sol_emb,
                            backtick_end_emb,
                            double_enters_emb,
                            *all_other_sols,
                            correct_solution_emb,
                            backtick_start_emb,
                            lets_think_step_by_step_emb,
                        ]
                    )
                else:
                    res = torch.concat(
                        [
                            debate_prompt_emb,
                            *all_other_sols,
                            your_solution_emb,
                            backtick_start_emb,
                            _my_sol_emb,
                            backtick_end_emb,
                            double_enters_emb,
                            correct_solution_emb,
                            backtick_start_emb,
                            lets_think_step_by_step_emb,
                        ]
                    )
            else:  ## wizardmath
                res = torch.concat(
                    [
                        instruct_wizardmath,
                        *all_other_sols,
                        your_solution_emb,
                        _my_sol_emb,
                        double_enters_emb,
                        debate_prompt_emb,
                        correct_solution_emb,
                        lets_think_step_by_step_emb,
                    ]
                )

            return res

        backtick_start = torch.tensor([self.sep_token_ids["backtick_start"]], device=self.device)  ## : ```...
        backtick_end = torch.tensor([self.sep_token_ids["backtick_end"]], device=self.device)  ## ...```\n

        backtick_start_emb = self.emb_table(backtick_start)
        backtick_end_emb = self.emb_table(backtick_end)
        double_enters_emb = self.emb_table(self.sep_token_ids["double_enters"])
        your_solution_emb = self.emb_table(self.sep_token_ids["your_solution"])
        correct_solution_emb = self.emb_table(self.sep_token_ids["correct_solution"])
        lets_think_step_by_step_emb = self.emb_table(self.sep_token_ids["lets_think_step_by_step"])
        if self.engine.startswith("wizardmath"):
            instruct_wizardmath = self.emb_table(self.sep_token_ids["instruct_wizardmath"])

        embs = [get_debate_promp_emb(p, o, m) for p, o, m in zip(prompts, other_sol_embs_list, my_sol_embs)]
        return embs  ## List of tensor, each has a shape of (1, t, d)

    def remove_promp_emb(self, embs: Tensor, promp_lens: List) -> List[Tensor]:
        res = [emb[l:,] for emb, l in zip(embs, promp_lens)]
        return res

    @torch.inference_mode()
    def generate_output_embs(
        self,
        prompts: List[str],
        temperature: float,
        is_debate: bool,
        my_sol_embs: Optional[List[Tensor]],
        other_sol_embs_list: Optional[List[List[Tensor]]],
        top_p_emb: float,
        l2_norm: float,
        top_k_emb: int,
        early_stop: bool,
        convert_to_cpu: bool,
    ) -> Dict:
        """
        my_sol_embs: Each element is a tensor
        other_sol_embs_list: Each element is a list of (n_agents-1) tensors
        """

        if not is_debate:
            prompt_embs = self._text_to_emb(prompts, is_bos=True)
        else:
            prompt_embs = self._get_debate_prompt_embs(
                prompts, other_sol_embs_list, my_sol_embs  ## type: ignore
            )  ## type: ignore
            prompts_nearest_neighbour = []
            for p in prompt_embs:
                prompt_txt, _ = self._decode_embs_nearest_neighbour(
                    rearrange(p, "t d-> () t d"), [0], self.context_length, early_stop=False
                )

                prompts_nearest_neighbour.append(prompt_txt[0])
            prompts = prompts_nearest_neighbour

        prompt_lens = [len(t) for t in prompt_embs]

        all_token_embs, human_readable_texts, token_embs_llama1_vs_llama2 = self._generate_embs(
            prompt_embs=prompt_embs,
            temperature=temperature,
            top_p_emb=top_p_emb,
            l2_norm=l2_norm,
            top_k_emb=top_k_emb,
        )

        nearest_neighbor_texts, stop_idxs = self._decode_embs_nearest_neighbour(
            all_token_embs, prompt_lens, self.max_new_tokens, early_stop=early_stop
        )

        if token_embs_llama1_vs_llama2 is not None:  ## llama 1 vs llama 2
            other_generated_token_embs = self.remove_promp_emb(token_embs_llama1_vs_llama2, prompt_lens)
            other_generated_token_embs = [
                emb[:stop] for emb, stop in zip(other_generated_token_embs, stop_idxs)
            ]
        else:
            other_generated_token_embs = None
        generated_token_embs = self.remove_promp_emb(all_token_embs, prompt_lens)
        generated_token_embs = [emb[:stop] for emb, stop in zip(generated_token_embs, stop_idxs)]

        ## Map the last embedding to A, B, C, D if the dataset is mmlu
        if not self.no_convert_ans_choice:
            nearest_neighbor_texts = self._nearest_token_choice(
                nearest_neighbor_texts, generated_token_embs, vector=True
            )

        if convert_to_cpu:
            generated_token_embs = [emb.cpu() for emb in generated_token_embs]
            if other_generated_token_embs is not None:
                other_generated_token_embs = [emb.cpu() for emb in other_generated_token_embs]
        return {
            "emb": generated_token_embs,
            "other_emb": other_generated_token_embs,
            "nearest_neighbor_texts": nearest_neighbor_texts,
            "human_readable_texts": human_readable_texts,
            "prompt": prompts,
        }

    def _sample_top_p(self, probs: Tensor, p: float):
        """
        probs: (b, vocab_size)
        p: scala number
        """
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)

        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    def _get_probs_after_top_p_k_emb(self, probs: Tensor, top_p_emb: float, top_k_emb: int) -> Tensor:
        """
        probs: (b, vocab_size).
        top_k_emb: keep top k with the highest probs
        """
        if top_k_emb != -1:
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            mask_out = torch.argsort(probs_idx) >= top_k_emb
            probs[mask_out] = 0.0
            probs.div_(probs.sum(dim=-1, keepdim=True))  ## re-norm

        if top_p_emb < 1.0:
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p_emb

            gather_idxs = torch.argsort(probs_idx)
            mask_out = torch.gather(mask, -1, gather_idxs)
            probs[mask_out] = 0.0  ## mask out tokens with tiny probabilities
            probs.div_(probs.sum(dim=-1, keepdim=True))  ## re-norm

        return probs
