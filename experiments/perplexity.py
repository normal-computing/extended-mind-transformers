"""Implementation of perplexity experiment for extended llama models."""

import json
import os
import pickle
from typing import Optional

import torch
import wandb
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
from tqdm import tqdm
from transformers import AutoTokenizer

from experiments.utils import save_config
from src.llama.modeling import ExtendedLlamaConfig, ExtendedLlamaForCausalLM
from src.mpt.modeling import ExtendedMptConfig, ExtendedMptForCausalLM


class PerplexityExperiment:
    """
    Evaluate langauge model perplexity
    """

    def __init__(self, config: FrozenConfigDict):
        self.config = ConfigDict(config)  # thaw config
        self.experiment_log_dir = config.experiment_log_dir

        self.input_lengths = config["input_lengths"]
        self.persist_cache = config["persist_cache"]
        self.devices = config["devices"]
        self.strategy = config["strategy"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_pretrained_model_name_or_path"]
        )
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
        assert config["model_architecture"] in ["llama", "mpt"]

        if config["model_architecture"] == "llama":
            rope_scaling = config["model_config"].pop("rope_scaling", None)
            self.model = ExtendedLlamaForCausalLM.from_pretrained(
                config["pretrained_model_name_or_path"],
                config=ExtendedLlamaConfig(
                    rope_scaling=dict(rope_scaling)
                    if rope_scaling is not None
                    else None,
                    **config["model_config"],
                ),
            ).to(self.devices[0])
        elif config["model_architecture"] == "mpt":
            self.model = ExtendedMptForCausalLM.from_pretrained(
                config["pretrained_model_name_or_path"],
                config=ExtendedMptConfig(**config["model_config"]),
            ).to(self.devices[0])

    def truncated_pass(
        self, inputs: torch.Tensor, experiment_input_length: int, stride: int,
    ):
        """
        Pass through single document, truncating inputs to max length
        """
        max_input_length = self.config.seq_len_trained
        nlls = []
        prev_end_loc = 0
        seq_len = inputs.size(1)
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + experiment_input_length, seq_len)
            subseq = inputs[:, begin_loc:end_loc]
            subseq = subseq[:, -max_input_length:]  # only use last max_length tokens

            assert subseq.size(1) <= self.config.max_inference_seq_len

            targets = subseq.clone()
            trg_len = end_loc - prev_end_loc
            targets[:, :-trg_len] = -100  # only compute loss on last trg_len tokens

            assert torch.sum(targets != -100) <= max_input_length

            with torch.no_grad():
                output = self.model(
                    input_ids=subseq.to(self.devices[0]),
                    use_active_externalism=False,
                    labels=targets,
                )
                nlls.append(output.loss)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        return nlls

    def naive_pass(
        self, inputs: torch.Tensor, experiment_input_length: int, stride: int,
    ):
        """
        Pass through single document, arbitrarily long inputs
        """
        if self.config.max_inference_seq_len < experiment_input_length:
            raise ValueError(
                "Naive methods require config updates before initialization."
            )

        max_input_length = self.config.seq_len_trained

        nlls = []
        prev_end_loc = 0
        seq_len = inputs.size(1)
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + experiment_input_length, seq_len)
            subseq = inputs[:, begin_loc:end_loc]

            assert subseq.size(1) <= self.config.max_inference_seq_len

            targets = subseq.clone()
            trg_len = end_loc - prev_end_loc
            targets[:, :-trg_len] = -100

            targets[
                :,
                : (
                    subseq.size(1) - max_input_length
                    if subseq.size(1) - max_input_length > 0
                    else 0
                ),
            ] = -100
            assert torch.sum(targets != -100) <= max_input_length

            with torch.no_grad():
                output = self.model(
                    input_ids=subseq.to(self.devices[0]),
                    use_active_externalism=False,
                    labels=targets,
                )
                nlls.append(output.loss)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        return nlls

    def extended_pass(
        self,
        inputs: torch.Tensor,
        experiment_input_length: int,
        stride: int,
        persist_cache: bool = False,
        verbose: bool = False,
    ):
        """
        Pass through single document, using active externalism
        """
        max_input_length = self.config.seq_len_trained

        nlls = []
        prev_end_loc = 0
        seq_len = inputs.size(1)
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + experiment_input_length, seq_len)
            subseq = inputs[:, begin_loc:end_loc]

            targets = subseq.clone()
            trg_len = end_loc - prev_end_loc
            targets[
                :, :-trg_len
            ] = -100  # only used if subseq - config_max_length < stride

            targets = targets[:, -max_input_length:]  # truncate targets, cache inputs

            assert torch.sum(targets != -100) <= max_input_length

            to_cache = subseq[
                :,
                : (
                    subseq.size(1) - max_input_length
                    if subseq.size(1) - max_input_length > 0
                    else 0
                ),
            ]
            if verbose:
                print(
                    f"Inputs are {subseq.size(1)} long. Caching {to_cache.size(1)} tokens."
                )

            subseq = subseq[:, -max_input_length:]

            if persist_cache:
                to_cache = torch.cat([inputs[:, :begin_loc], to_cache], dim=1)

            if to_cache.numel() > 0:
                self.model.memory_ids = to_cache
                self.model.memories = None
            else:
                self.model.clear_memory()
                if verbose:
                    print(
                        "Not persisting memories, no new memories. Memory has been emptied."
                    )

            if verbose:
                print(
                    f"""Cache is {self.model.memory_ids.size(1) 
                    if self.model.memory_ids is not None else 0} long."""
                )

            with torch.no_grad():
                output = self.model(
                    input_ids=subseq.to(self.devices[0]),
                    use_active_externalism=True,
                    labels=targets,
                )
                nlls.append(output.loss)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        return nlls

    def run_experiment(self, dataset_path: str, verbose: bool = True, **kwargs):
        """
        Evaluate perplexity on dataset for all given input lengths.
        Experiment input length: length of input to use for each pass
        Stride: length of overlap between passes
        """

        with open(dataset_path, "rb") as file:
            dataset = json.load(file)

        raw_data = wandb.Artifact(
            dataset_path.split("/")[-1],
            type="dataset",
        )

        with raw_data.new_file("dataset.json", mode="wb") as file:
            torch.save(dataset, file)
        wandb.log_artifact(raw_data)

        forward_map = {
            "naive": self.naive_pass,
            "extended": self.extended_pass,
            "truncate": self.truncated_pass,
        }
        forward_pass = forward_map[self.strategy]
        kwargs["persist_cache"] = self.persist_cache
        kwargs["verbose"] = self.config.verbose

        results = []
        results_table = wandb.Table(
            columns=["input_length", "id", "title", "avg_nll", "ppl"]
        )
        for b, experiment_input_length in enumerate(self.input_lengths):
            avg_nlls = ()
            for _, sample in tqdm(enumerate(dataset)):
                inputs = sample["inputs"]
                input_ids = self.tokenizer(inputs, return_tensors="pt")["input_ids"]

                loss = forward_pass(
                    input_ids,
                    experiment_input_length=experiment_input_length,
                    stride=512,
                    **kwargs,
                )

                nll = torch.stack(loss).mean()
                ppl = torch.exp(nll)

                wandb.log(
                    {
                        "Sample Avg NLL": nll,
                        "input_length": experiment_input_length,
                    }
                )
                wandb.log(
                    {
                        "Sample PPL": ppl,
                        "input_length": experiment_input_length,
                    }
                )

                avg_nlls += (nll,)

                results += [
                    {
                        "input_length": experiment_input_length,
                        "id": sample["idx"],
                        "title": sample["heading"],
                        "avg_nll": nll.item(),
                        "ppl": ppl.item(),
                    }
                ]

                results_table.add_data(
                    experiment_input_length,
                    sample["idx"],
                    sample["heading"],
                    nll.item(),
                    ppl.item(),
                )

            avg_nll = torch.stack(avg_nlls).mean()
            avg_ppl = torch.exp(avg_nll)

            if verbose:
                print(
                    f"Average PPL for input length {experiment_input_length}: {avg_ppl}"
                )

            wandb.log({"Avg NLL": avg_nll, "input_length": experiment_input_length})
            wandb.log({"Avg PPL": avg_ppl, "input_length": experiment_input_length})

            self.save_results(results, metadata=experiment_input_length, checkpoint=b)

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
