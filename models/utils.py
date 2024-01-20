import json
from zoneinfo import ZoneInfo
from typing import Optional
import torch
import gc
from typing import Dict
import yaml


def type_list(type_: str):
    if type_ == "int":
        return lambda s: [int(x) for x in s.split(",")]
    elif type_ == "float":
        return lambda s: [float(x) for x in s.split(",")]
    elif type_ == "str":
        return lambda s: [x for x in s.split(",")]
    else:
        raise NotImplementedError()


def ensure_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]


def maybe_duplicate(x, n_elements):
    x = ensure_list(x)
    if len(x) < n_elements:
        res = (x * n_elements)[:n_elements]
    else:
        res = x
    return res


def datetime_now(time_format: Optional[str] = None) -> str:
    from datetime import datetime

    if time_format is None:
        time_format = "%Y-%b-%d--%H-%M-%S"
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime(time_format)


def get_model_path(model_name, hdfs: bool = False):
    with open("config.yml", "r") as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)

    if not hdfs:
        model_name = model_name.replace("_expert", "").replace("_dummy", "")
        if model_name.startswith("Llama-2"):
            model_path = yaml_config["llama2_model_path"].format(model_name=model_name)
        elif model_name.startswith("llama"):
            model_path = yaml_config["llama1_model_path"].format(model_name=model_name)
        elif model_name.startswith("falcon"):
            model_path = yaml_config["falcon_model_path"].format(model_name=model_name)
        elif model_name.startswith("dummy") or model_name.startswith("expert"):
            return ""
        elif model_name.startswith("mpt"):
            model_path = yaml_config["mpt_model_path"].format(model_name=model_name)
        elif model_name.startswith("WizardMath"):
            model_path = yaml_config["wizardmath_model_path"].format(model_name=model_name)
        else:
            raise ValueError(f"Invalid model name {model_name}")
    else:
        return model_name
    return model_path


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_gpu_mem_all() -> None:
    ## get all gpu available
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        free_gb = get_gpu_mem(cuda=f"cuda:{i}")
        print(f"\tdevice: {i+1}/{n_gpus}, avail mem: {free_gb}GB")


def get_gpu_mem(cuda="cuda:0") -> str:
    free, total = torch.cuda.mem_get_info(device=cuda)
    free_gb, total_gb = free / 1024**3, total / 1024**3
    return f"{round(free_gb, 2)}/{round(total_gb, 2)}"


def clear_gpu_mem(verbose: bool = False):
    if verbose:
        print(f"mem available before clearing:")
        get_gpu_mem_all()

    gc.collect()
    torch.cuda.empty_cache()

    if verbose:
        print(f"mem available after clearing:")
        get_gpu_mem_all()


def read_json(data_path: str) -> Dict:
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def duplicate_temp(n_rounds, n_agents, temp):
    ## update temperature_1 in new args
    temperatures = [-1] * n_rounds * n_agents
    for d in range(n_agents):
        temperatures[d * n_rounds : (d + 1) * n_rounds] = [temp[d]] * n_rounds
    return temperatures
