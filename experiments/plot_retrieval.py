import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def load(data_path, load_json=False):
    if not load_json:
        with open(data_path, "rb") as f:
            data = pickle.load(f)
    else:
        with open(data_path) as f:
            data = json.load(f)

    return pd.DataFrame(data)

def create_model_positions(x_pos, bar_width, evaluations):
    num_evaluations = len(evaluations)
    positions = []

    for i in range(num_evaluations):
        offset = (i - (num_evaluations - 1) / 2) * bar_width
        position = x_pos + offset
        positions.append([position])
    return positions

def simple_bar_plot(data, evaluations:dict, colors=["#B3E5FC", "#0D47A1", "#64B5F6", "#0077B6"]):
    frames = {}
    for name, df in evaluations.items():
        df["split"] = df["split"].apply(lambda x: int(x.replace("k", "")) * 1000)
        frames[name] = (
            df[["id", "split", "eval", "context", "n_replacements"]]
        )
    data["split"] = data["split"].apply(lambda x: int(x.replace("k", "")) * 1000)
    
    plt.figure(figsize=(10, 8))
    # Create bar graphs
    bar_width = 0.2
    x_pos = np.arange(len(set(data["split"].values))) 
    positions = create_model_positions(x_pos, bar_width, evaluations)
    for name, df in frames.items():
        data_to_plot = list((df.groupby('split')["eval"].sum() / data.groupby('split').size()).values)
        pos = positions.pop(0)
        plt.bar(pos[0], data_to_plot, width=bar_width, align='center', label=name, color = colors.pop(0) )

    plt.xlabel('Document length')
    plt.ylabel('% Correct QA')
    plt.title('Question Answer Results')
    plt.xticks(x_pos,['2k', '4k', '8k', '16k'])
    plt.legend()
    
    # Display plot
    plt.tight_layout()
    plt.savefig('bar_graph_retrieval.png')

def process(data, evaluations: dict):
    frames = []
    for name, df in evaluations.items():
        df["split"] = df["split"].apply(lambda x: int(x.replace("k", "")) * 1000)
        frames.append(
            df[["id", "split", "eval", "context", "n_replacements"]].rename(
                columns={"eval": name}
            )
        )

    data["split"] = data["split"].apply(lambda x: int(x.replace("k", "")) * 1000)

    frame = frames[0]
    for addtl_frame in frames[1:]:
        frame = frame.merge(
            addtl_frame, on=["id", "split", "context", "n_replacements"]
        )

    # Get bins using Qcut
    _, bins = pd.qcut(frame["n_replacements"], q=70, retbins=True, duplicates="drop")
    bins = [bin.round() for bin in bins]

    # Cut as categorical
    frame["Binned Num. Replacements"] = pd.cut(
        frame["n_replacements"], bins=bins, right=True
    )
    pivot_table = pd.pivot_table(
        frame,
        values=evaluations.keys(),
        index=["split", "Binned Num. Replacements"],
        aggfunc="sum",
    ).reset_index()
    pivot_table = pivot_table.pivot(
        index="Binned Num. Replacements", columns="split", values=evaluations.keys()
    ).fillna(0)

    data["Binned Num. Replacements"] = pd.cut(
        data["n_replacements"], bins=bins, right=True
    )
    data_ptable = pd.pivot_table(
        data,
        values="context",
        index=["split", "Binned Num. Replacements"],
        aggfunc="count",
    ).reset_index()
    data_ptable = data_ptable.pivot(
        index="Binned Num. Replacements", columns="split", values="context"
    ).fillna(0)

    return pivot_table / data_ptable

def plot(pivot_table, model_names, save_path="retrieval.png", monotone=True):
    if monotone:     # Create a custom colormap from light to dark blue
        cmap = LinearSegmentedColormap.from_list(
            "custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"]
        )
    else:
        cmap = LinearSegmentedColormap.from_list(
                "custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"]
            )
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#B3E5FC", "#64B5F6", "#0077B6", "#0D47A1"])
    
    pivot_table.columns = pivot_table.columns.droplevel(0)
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    sns.heatmap(
        pivot_table, fmt="g", cmap=cmap, cbar_kws={"label": "% Correct Retrievals"}
    )

    plt.xlabel(" ")  # X-axis label
    plt.ylabel("Number of Fact Appearances")  # Y-axis label

    num_models = len(model_names)
    model_divisions = [4 * i for i in range(1, num_models + 1)]
    for x in model_divisions:
        plt.axvline(x=x, color="white", linestyle="-", lw=20, zorder=2)

    model_positions = [2.25 + (4 * i) for i in range(num_models)]
    for pos, model_name in zip(model_positions, model_names):
        plt.text(
            pos,
            19,
            model_name,
            ha="center",
            va="center",
            fontweight="bold" if model_name == "Extended Mind" else "normal",
            fontdict={'size': 12},
            zorder=3
        )

    # Add a box around the first two models
    rect = plt.Rectangle((.01, -.01), 7.85, pivot_table.shape[0], fill=False, edgecolor="red", linewidth=5, zorder=4)
    plt.gca().add_patch(rect)

    # Add text above the rectangle
    plt.text(
        3.95, 
        pivot_table.shape[0] - 17.5, 
        "Not Finetuned", 
        ha="center", 
        va="center", 
        fontweight="bold", 
        fontdict={'size': 12}, 
        zorder=5 
)
    #Title
    plt.text(
    17.5 //2, 
    pivot_table.shape[0] - 18.5, 
    "% Correct Fact Retrievals by Document Length", 
    ha="center", 
    va="center", 
    fontdict={'size': 14} 
)
    plt.xticks(rotation=45) 
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    MODEL_1 = "./runs/retrieval/date-and-date/path/to/checkpoint.pkl"
    MODEL_2 = "./runs/retrieval/date-and-date/path/to/checkpoint.pkl"
    MODEL_3 = "./runs/retrieval/date-and-date/path/to/checkpoint.pkl"
    DATA = "./experiments/data/wikiqa-edited.json"

    model1 = load(MODEL_1)
    model2 = load(MODEL_2)
    model3 = load(MODEL_3)
    data = load(DATA, load_json=True)

    evaluations = {
        "Model 1": model1,
        "Model 2": model2,
        "Model 3": model3,
    }

    pivot = process(data, evaluations)
    plot(pivot, evaluations.keys())

    simple_bar_plot(data, evaluations)
