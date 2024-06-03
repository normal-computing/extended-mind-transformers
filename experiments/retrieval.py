import json
import os
import pickle
from typing import Optional

import omegaconf
import torch
import wandb
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
from tqdm import tqdm

from experiments.chat_prompting import llama_chat_prompt, mpt_chat_prompt
from experiments.closed_generation import query_claude, query_gpt
from experiments.utils import save_config
from experiments.rag import get_chunks
from experiments.rag2 import augmented_generate

class RetrievalExperiment:
    def __init__(self, config: FrozenConfigDict):
        self.config = ConfigDict(config)  # thaw config
        config_as_dict = self.config.to_dict()
        wandb.init(
            project=config["experiment_name"],
            config=config_as_dict,
            notes=config["notes"] if "notes" in config else None,
            tags=config["tags"] if "tags" in config else None,
        )
        self.config.wandb_id = wandb.run.id
        self.config.wandb_name = wandb.run.name
        save_config(
            self.config.to_dict(), config["experiment_log_dir"] + "/config.yaml"
        )

        self.experiment_log_dir = config.experiment_log_dir
        self.devices = config["devices"] if "devices" in config else None
        self.chat_model = config["chat_model"]
        self.model_extended = config["model_extended"]
        self.use_rag2 = config["use_rag2"]
        if self.use_rag2:
            self.rag2_config = config["rag2_config"]
        self.use_rag = config["use_rag"]
        if self.use_rag:
            self.rag_config = config["rag_config"]
        self.dynamic_topk = config["dynamic_topk"]
        self.n_tokens = config["n_tokens"]
        self.one_shot = config["one_shot"] if not self.chat_model else None

        #Only case we exclude the document is when we are using EMTS alone
        self.exclude_document = self.model_extended and not self.use_rag

        if config["model_location"] == "local":  # Generating locally
            assert config["model_architecture"] in ["llama", "mpt"]

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

            self.model_kwargs = {}
            if config['precision'] != 'float8':
                self.model_kwargs.update({'torch_dtype': getattr(torch, config['precision'])})
            else:
                self.model_kwargs.update({'load_in_8bit':True})
            if config["device_map"] is not None:
                self.model_kwargs.update({'device_map':config["device_map"]})

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
                    config=model_config if "model_config" in config else None,
                    trust_remote_code=True,
                    **self.model_kwargs
                )

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
                    **self.model_kwargs
                )
            elif config["model_architecture"] == "mpt":
                from emts_clean.src.mpt.modeling import (
                    ExtendedMptConfig,
                    ExtendedMptForCausalLM,
                )

                self.model = ExtendedMptForCausalLM.from_pretrained(
                    config["pretrained_model_name_or_path"],
                    config=ExtendedMptConfig(**config["model_config"]),
                    **self.model_kwargs
                )
            if 'device_map' not in self.model_kwargs:
                self.model.to(self.devices[0])

        elif config["model_provider"] == "openai":
            self.generate = query_gpt
        elif config["model_provider"] == "anthropic":
            self.generate = query_claude

    def prepare_prompt(self, prompt, question, document):
        if self.config["model_provider"] == "openai" and self.chat_model:
            inputs = [
                {
                    "role": "system",
                    "content": prompt,
                },
                {"role": "user", "content": document},
                {
                    "role": "user",
                    "content": question,
                },
            ]
        if self.config["model_provider"] == "anthropic" and self.chat_model:
            inputs = (
                prompt,
                [
                    {"role": "user", "content": " ".join([document, question])},
                ],
            )
        if self.config["model_location"] != "local" and not self.chat_model:
            raise NotImplementedError(
                "Currently only support Chat versions of GPT and Claude"
            )

        if self.config["model_location"] == "local" and self.chat_model:
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = (
                    "\n".join([prompt, document, question])
                    if not self.exclude_document
                    else "\n".join([prompt, question])
                )
                inputs = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            elif self.config["model_architecture"] == "llama":
                prompt = (
                    llama_chat_prompt(
                        [
                            {"role": "system", "content": prompt},
                            {
                                "role": "user",
                                "content": "\n".join([document, question]),
                            },
                        ]
                    )
                    if not self.exclude_document
                    else llama_chat_prompt(
                        [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": question},
                        ]
                    )
                )
                inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
            elif self.config["model_architecture"] == "mpt":
                prompt = (
                    mpt_chat_prompt(
                        [
                            {"role": "system", "content": prompt},
                            {
                                "role": "user",
                                "content": "\n".join([document, question]),
                            },
                        ]
                    )
                    if not self.exclude_document
                    else mpt_chat_prompt(
                        [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": question},
                        ]
                    )
                )
                inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
            else:
                raise NotImplementedError(
                    "Please provide the chat prompt formatting for your model if not given by the tokenizer."
                )
        elif self.config["model_location"] == "local" and not self.chat_model:
            question = question + "\nAnswer:"
            prompt = (
                "\n".join([self.one_shot, prompt, document, question])
                if not self.exclude_document
                else "\n".join([self.one_shot, prompt, question])
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        return inputs

    def run_experiment(self, dataset_path: str):
        with open(dataset_path, "rb") as file:
            dataset = json.load(file)

        raw_data = wandb.Artifact(
            dataset_path.split("/")[-1],
            type="dataset",
        )
        with raw_data.new_file("dataset.json", mode="wb") as file:
            torch.save(dataset, file)
        wandb.log_artifact(raw_data)

        results = []
        total_correct = 0
        results_table = wandb.Table(columns=["idx", "split", "text", "eval"])
        generate_kwargs = {}
        for idx, sample in tqdm(enumerate(dataset)):
            if self.model_extended:
                self.model.memory_ids = self.tokenizer(sample["context"])["input_ids"]
            if self.dynamic_topk:
                topk = max(min(round(len(self.model.memory_ids[0])/500), 12), 4)
                generate_kwargs.update({'topk':topk})
            if self.use_rag:
                chunks, _ = get_chunks(sample["context"], sample["question"], self.tokenizer, n_chunks=self.rag_config["n_chunks"],
                                        tokens_per_chunk=self.rag_config["tokens_per_chunk"], chunk_overlap=self.rag_config["chunk_overlap"])
                document = "\n".join(chunks)
            else:
                document = sample["context"]

            inputs = self.prepare_prompt(
                prompt=sample["prompt"],
                question=sample["question"],
                document=document,
            )

            if self.config.model_location == "local":
                if self.use_rag2:
                    generate_kwargs.update(self.rag2_config)
                    generate_kwargs.update({'dynamic_topk': self.dynamic_topk})
                    text, out, _ = augmented_generate(
                        self.model,
                        self.tokenizer,
                        inputs,
                        **generate_kwargs,
                    )
                else:
                    out = self.model.generate(
                        inputs.to(self.model.device),
                        max_length=self.n_tokens + inputs.size(-1),
                        generation_config=self.generation_config,
                        **generate_kwargs,
                        
                        #necessary for llama3
                        do_sample=False,
                        bos_token_id=128000,
                        eos_token_id=128001,
                        pad_token_id=0
                        #####
                    )
                    text = self.tokenizer.decode(out[0][-self.n_tokens :])
            else:
                key_name = f'{self.config["model_provider"]}_key'
                text = self.generate(
                    messages=inputs,
                    api_key=os.environ[key_name],
                    model_name=self.config["model_name"],
                    max_tokens=self.n_tokens,
                )

            evaluation = int(sample["answer"].lower() in text.lower())

            if self.model_extended:
                self.model.clear_memory()

            results_table.add_data(idx, sample["split"], text, evaluation)
            result = {"id": idx, "split": sample["split"], "text": text, "eval": evaluation}
            results.append(sample | result)
            self.save_results(results, checkpoint=sample["split"])
            total_correct += result["eval"]

        wandb.log({"total_correct": total_correct})
        wandb.log({"results-table": results_table})
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
        wandb.finish()

        return results
