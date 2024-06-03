import json
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load(data_path, load_json=False):
    if not load_json:
        with open(data_path, "rb") as f:
            data = pickle.load(f)
    else:
        with open(data_path) as f:
            data = json.load(f)

    return pd.DataFrame(data)


def remove_outliers(frame, col="avg_nll", n_std=2):
    mean = frame[col].mean()
    std = frame[col].std()
    criteria = (frame[col] > mean - n_std * std) & (frame[col] < mean + n_std * std)
    return frame.loc[criteria]


def process(evaluations, drop_outliers=False):
    frames = []
    for name, df in evaluations.items():
        df = (
            df.groupby("input_length").apply(remove_outliers).reset_index(drop=True)
            if drop_outliers
            else df
        )
        frames.append(
            df.rename(columns={"ppl": f"{name}_ppl", "avg_nll": f"{name}_avg_nll"})
        )

    frame = frames[0]
    for addtl_frame in frames[1:]:
        frame = frame.merge(addtl_frame, on=["input_length", "id", "title"])

    names = list(evaluations.keys())
    grouped = pd.DataFrame(
        frame.groupby("input_length")[f"{names[0]}_avg_nll"]
        .apply(lambda x: math.exp(np.mean(x)))
        .sort_index()
    )
    for name in names[1:]:
        grouped = grouped.join(
            frame.groupby("input_length")[f"{name}_avg_nll"]
            .apply(lambda x: math.exp(np.mean(x)))
            .sort_index()
        )

    return grouped


def plot(group, names, crop="naive", save_path="perplexity.png"):
    if crop == "naive":
        group = group[group.index <= 4608]
    elif crop == "truncate":
        group = group[group.index >= 1024]

    fig, ax = plt.subplots()
    for name in names:
        ax.plot(group.index, group[f"{name}_avg_nll"], label=name)

    ax.set(xlabel="Input Length", ylabel="Perplexity (<-- lower is better)")
    ax.set_title("MPT-7b Perplexity")
    ax.legend()
    plt.savefig(save_path)  # Save the plot


if __name__ == "__main__":
    TRUNCATE = "./runs/perplexity/date-and-date/path/to/checkpoint.pkl"
    NAIVE = "./runs/perplexity/date-and-date/path/to/checkpoint.pkl"
    EXTENDED_MIND = "./runs/perplexity/date-and-date/path/to/checkpoint.pkl"

    truncate = load(TRUNCATE)
    naive = load(NAIVE)
    extended_mind = load(EXTENDED_MIND)

    evaluations = {'Naive': naive, 'Truncate':truncate, 'Extended Mind': extended_mind}

    grouped = process(evaluations, drop_outliers=True)
    plot(grouped, evaluations.keys(), crop='truncate')