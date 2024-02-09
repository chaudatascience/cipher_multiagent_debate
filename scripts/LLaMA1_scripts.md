Note on NLD's and CIPHER's commands

For [Natural Language Debate (NLD)](https://arxiv.org/abs/2305.14325), prompt path (`--debate_prompt_paths`) does **not** contain "_vector_language" in the file name, while for CIPHER, it does. 

Also, for CIPHER, we need to add flag `-v` to the command to switch it to embedding communication mode.

For example, NLD:

`--debate_prompt_paths prompts_v2/mmlu/debate_high_school_mathematics_2debaters_v1.txt` 

to convert the command to run CIHPHER:

`--debate_prompt_paths prompts_v2/mmlu/debate_high_school_mathematics_2debaters_vector_language_v1.txt -v`

The rest of params are the same for both methods.



# 1. GSM8K - LLaMA1
## 1.1. NLD (Human language debate)
python3 run_debate.py -p 5 -d gsm8k  --debaters llama_65B,llama_65B -b 8 --initial_prompt_paths prompts_v2/gsm8k/init_question_3shot_v3.txt --debate_prompt_paths prompts_v2/gsm8k/debate_2debaters_v1.txt --temperature_max 2.0 --max_new_tokens 400 --n_rounds 3 --data_path data/gsm/test_gsm8k_full.jsonl --n_questions 200 --n_gpus_per_actor 4 --n_ray_actors 2 --temperatures 0.1,0.2


## 1.2. CIPHER 

python3 run_debate.py -p 5 -d gsm8k --debaters llama_65B,llama_65B -b 8 --initial_prompt_paths prompts_v2/gsm8k/init_question_3shot_v3.txt --debate_prompt_paths prompts_v2/gsm8k/debate_2debaters_vector_language_v1.txt -v --temperature_max 2.0 --max_new_tokens 400 --n_rounds 3 --data_path data/gsm/test_gsm8k_full.jsonl --n_questions 200 --n_gpus_per_actor 4 --n_ray_actors 2 --temperatures 0.25,0.85

## 1.3. Majority voting
python3 run_debate.py -p 5 -d gsm8k --debaters llama_65B,llama_65B -b 8 --initial_prompt_paths prompts_v2/gsm8k/init_question_3shot_v3.txt --debate_prompt_paths prompts_v2/gsm8k/debate_2debaters_v1.txt --temperature_max 2.0 --max_new_tokens 400 --n_rounds 1 --n_sols_each_ques 5 --data_path data/gsm/test_gsm8k_full.jsonl --n_questions 200 --n_gpus_per_actor 4 --n_ray_actors 2 --temperatures 0.8,0.8

## 1.4. Single answer
python3 run_debate.py -p 3 -d gsm8k --debaters llama_65B,llama_65B -b 8 --initial_prompt_paths prompts_v2/gsm8k/init_question_3shot_v3.txt --debate_prompt_paths prompts_v2/gsm8k/debate_2debaters_v1.txt --temperature_max 2.0 --max_new_tokens 400 --n_rounds 1 --n_sols_each_ques 1 --data_path data/gsm/test_gsm8k_full.jsonl --n_questions 200 --n_gpus_per_actor 4 --n_ray_actors 2 --temperatures 0.2,0.2

# 2. Arithmetic
Adding “multiagent_debate/data/arithmetic/test_seed2024.jsonl”

## 2.1. NLD
python3 run_debate.py -p 5 -d arithmetic --debaters llama_65B,llama_65B -b 20 --initial_prompt_paths prompts_v2/arithmetic/init_prompt.txt --debate_prompt_paths prompts_v2/arithmetic/debate_2debaters_v1.txt --temperature_max 2.0 --max_new_tokens 120 --n_rounds 3 --data_path data/arithmetic/test_seed23.jsonl --n_questions 200 --n_gpus_per_actor 4 --n_ray_actors 1 --temperatures 0.08,0.2


## 2.2. CIPHER

python3 run_debate.py -p 5 -d arithmetic --debaters llama_65B,llama_65B -b 8 --initial_prompt_paths prompts_v2/arithmetic/init_prompt.txt --debate_prompt_paths prompts_v2/arithmetic/debate_2debaters_vector_language_v1.txt -v --temperature_max 2.0 --max_new_tokens 120 --n_rounds 3 --data_path data/arithmetic/test_seed23.jsonl --n_questions 200 --n_gpus_per_actor 4 --n_ray_actors 1 --temperatures 0.67,1.43

## 2.3. Majority voting
python3 run_debate.py -p 5 -d arithmetic --debaters llama_65B,llama_65B -b 20 --initial_prompt_paths prompts_v2/arithmetic/init_prompt.txt --debate_prompt_paths prompts_v2/arithmetic/debate_2debaters_v1.txt --temperature_max 2.0 --max_new_tokens 120 --n_rounds 1 --n_sols_each_ques 5 --data_path data/arithmetic/test_seed23.jsonl --n_questions 200 --n_gpus_per_actor 4 --n_ray_actors 1 --temperatures 0.8,0.8


## 2.4. Single answer
python3 run_debate.py -p 3 -d arithmetic --debaters llama_65B,llama_65B -b 20 --initial_prompt_paths prompts_v2/arithmetic/init_prompt.txt --debate_prompt_paths prompts_v2/arithmetic/debate_2debaters_v1.txt --temperature_max 2.0 --max_new_tokens 120 --n_rounds 1 --n_sols_each_ques 1 --data_path data/arithmetic/test_seed23.jsonl --n_questions 200 --n_gpus_per_actor 4 --n_ray_actors 1 --temperatures 0.4,0.4

# 3. Psychology 
## 3.1. NLD

python3 run_debate.py -p 5 -d mmlu --debaters llama_65B,llama_65B -b 12 --initial_prompt_paths prompts_v2/mmlu/init_professional_psychology_v1.txt --debate_prompt_paths prompts_v2/mmlu/debate_professional_psychology_2debaters_v1.txt --max_new_tokens 400 --n_rounds 3 --temperature_max 2.0 --n_questions 200 --data_path data/mmlu/test/professional_psychology_test.csv --n_gpus_per_actor 4 --n_ray_actors 2 --temperatures 0.3,0.4

## 3.2. CIPHER

python3 run_debate.py -p 5 -d mmlu --debaters llama_65B,llama_65B -b 12 --initial_prompt_paths prompts_v2/mmlu/init_professional_psychology_v1.txt --debate_prompt_paths prompts_v2/mmlu/debate_professional_psychology_2debaters_vector_language_v1.txt -v --max_new_tokens 400 --n_rounds 3 --temperature_max 2.0 --n_questions 200 --data_path data/mmlu/test/professional_psychology_test.csv --n_gpus_per_actor 4 --n_ray_actors 2 --temperatures 0.1,0.4

3.3. Majority voting
python3 run_debate.py -p 5 -d mmlu --debaters llama_65B,llama_65B -b 12 --initial_prompt_paths prompts_v2/mmlu/init_professional_psychology_v1.txt --debate_prompt_paths prompts_v2/mmlu/debate_professional_psychology_2debaters_v1.txt --max_new_tokens 400 --n_rounds 1 --n_sols_each_ques 5 --temperature_max 2.0 --n_questions 200 --data_path data/mmlu/test/professional_psychology_test.csv --n_gpus_per_actor 4 --n_ray_actors 2 --temperatures 0.6,0.6

3.4. Single answer
python3 run_debate.py -p 3 -d mmlu --debaters llama_65B,llama_65B -b 12 --initial_prompt_paths prompts_v2/mmlu/init_professional_psychology_v1.txt --debate_prompt_paths prompts_v2/mmlu/debate_professional_psychology_2debaters_v1.txt --max_new_tokens 400 --n_rounds 1 --n_sols_each_ques 1 --temperature_max 2.0 --n_questions 200 --data_path data/mmlu/test/professional_psychology_test.csv --n_gpus_per_actor 4 --n_ray_actors 2 --temperatures 0.2,0.2
