import json
import os
import pickle
import time
import omegaconf
import torch
from tqdm import tqdm
from typing import Optional
from ml_collections import ConfigDict, FrozenConfigDict

from experiments.utils import save_config


class TimingExperiment:
    def __init__(self, config: FrozenConfigDict):
        self.config = ConfigDict(config)
        save_config(
            self.config.to_dict(), config["experiment_log_dir"] + "/config.yaml"
        )

        self.experiment_log_dir = config.experiment_log_dir
        self.devices = config["devices"] if "devices" in config else None
        self.model_extended = config["model_extended"]
        self.n_tokens = config["n_tokens"]
        self.cache_context = config["cache_context"]

        assert config["model_architecture"] in ["llama", "mpt"]
        assert not (config["cache_context"] and config["model_extended"])

        transformers_version = config[
            "transformers_version"
        ]  # Different models may need different versions
        import subprocess
        import sys

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "transformers==" + transformers_version,
            ]
        )

        from transformers import AutoTokenizer, GenerationConfig

        self.tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_pretrained_model_name_or_path"]
        )

        self.generation_config = (
            GenerationConfig(config=config["generation_config"])
            if "generation_config" in config
            else None
        )
        model_dtype = torch.float16 if config["fp16"] else torch.float32

        if config["auto_model"]:
            from transformers import AutoConfig, AutoModelForCausalLM

            if "model_config" in config:
                if "rope_scaling" in config["model_config"]:
                    config["model_config"]["rope_scaling"] = dict(
                        config["model_config"].pop("rope_scaling")
                    )

                model_config = AutoConfig.from_pretrained(
                    config["pretrained_model_name_or_path"],
                    **omegaconf.OmegaConf.to_container(
                        config["model_config"], resolve=True
                    ),
                )

            self.model = AutoModelForCausalLM.from_pretrained(
                config["pretrained_model_name_or_path"],
                torch_dtype=model_dtype,
                config=model_config if "model_config" in config else None,
                trust_remote_code=True,
            ).to(self.devices[0])
        elif config["model_architecture"] == "llama":
            from emts_clean.src.llama.modeling import (
                ExtendedLlamaConfig,
                ExtendedLlamaForCausalLM,
            )

            rope_scaling = (
                config["model_config"].pop("rope_scaling", None)
                if "model_config" in config
                else None
            )
            self.model = ExtendedLlamaForCausalLM.from_pretrained(
                config["pretrained_model_name_or_path"],
                config=ExtendedLlamaConfig(
                    rope_scaling=dict(rope_scaling)
                    if rope_scaling is not None
                    else None,
                    **config["model_config"],
                )
                if "model_config" in config
                else None,
                torch_dtype=model_dtype,
            ).to(self.devices[0])
        elif config["model_architecture"] == "mpt":
            from emts_clean.src.mpt.modeling import (
                ExtendedMptConfig,
                ExtendedMptForCausalLM,
            )

            self.model = ExtendedMptForCausalLM.from_pretrained(
                config["pretrained_model_name_or_path"],
                config=ExtendedMptConfig(**config["model_config"]),
                torch_dtype=model_dtype,
            ).to(self.devices[0])

    def prepare_prompt(self, prompt, question, document):
        question = question + "\nAnswer:"
        inputs = (
            "\n".join([prompt, document, question])
            if not self.model_extended
            else "\n".join([prompt, question])
        )
        inputs = self.tokenizer(inputs, return_tensors="pt")["input_ids"]
        tokens_to_cache = self.tokenizer(
            "\n".join([prompt, document]), return_tensors="pt"
        )["input_ids"].shape[-1]
        return inputs, tokens_to_cache if self.cache_context else None

    def run_experiment(self, dataset_path: str):
        with open(dataset_path, "rb") as file:
            dataset = json.load(file)

        results = []
        for idx, sample in tqdm(enumerate(dataset)):
            if sample["split"] != "4k":  # Just use one split
                continue

            if self.model_extended:
                self.model.memory_ids = self.tokenizer(sample["context"])["input_ids"]

            inputs, tokens_to_cache = self.prepare_prompt(
                prompt=sample["prompt"],
                question=sample["question"],
                document=sample["context"],
            )
            times = []
            past_kvs = None
            for i in range(self.config.n_queries):
                s_time = time.time()
                out = self.model.generate(
                    inputs.to(self.model.device),
                    max_length=self.n_tokens + inputs.size(-1),
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                    past_key_values=past_kvs if self.cache_context else None,
                    attention_mask=torch.ones(
                        1, inputs.size(-1) + past_kvs[0][0].size(-2)
                    ).to(self.model.device)
                    if (self.cache_context and past_kvs is not None)
                    else None,
                )

                e_time = time.time()
                execution_time = e_time - s_time
                times.append(execution_time)

                past_kvs = (
                    [
                        (kv[0][:, :, :tokens_to_cache], kv[1][:, :, :tokens_to_cache])
                        for kv in out.past_key_values
                    ]
                    if self.cache_context and i == 0
                    else past_kvs
                )

                inputs = (
                    inputs[:, tokens_to_cache:]
                    if self.cache_context and i == 0
                    else inputs
                )

            if self.model_extended:
                self.model.clear_memory()

            result = {
                "id": idx,
                "split": sample["split"],
                "n_queries": self.config.n_queries,
                "times": times,
            }
            results.append(sample | result)
            self.save_results(results, checkpoint=sample["split"])

        return results

    def save_results(
        self, results, metadata: str = None, checkpoint: Optional[int] = None
    ):  # To do: make more specific than pickle
        """
        Save results as pickle file
        """
        folder = "final" if checkpoint is None else f"checkpoints/{checkpoint}"
        experiment_dir = os.path.join(self.experiment_log_dir, folder)
        os.makedirs(experiment_dir, exist_ok=True)
        result_file = os.path.join(
            experiment_dir, f"results-{metadata}.pkl" if metadata else "results.pkl"
        )
        with open(result_file, "wb") as f:
            pickle.dump(results, f)

    def run(self, dataset_path: str, **kwargs):
        """
        Run experiment
        """
        results = self.run_experiment(dataset_path, **kwargs)
        self.save_results(results)

        return results
