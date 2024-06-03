import pickle
import pandas as pd
import matplotlib.pyplot as plt


def load(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return pd.DataFrame(data)


def process(maxx, data):
    xaxis = [_ for _ in range(1, maxx + 1)]
    for i in xaxis:
        data[f"average-time-{i}"] = data["times"].apply(
            lambda x: sum(x[:i]) / len(x[:i])
        )
    return data


def plot(maxx, evalutations, save_path="timing.png"):
    xaxis = [_ for _ in range(1, maxx + 1)]

    data_pts = []
    names = []
    for name, data in evalutations.items():
        data_pts.append([data[f"average-time-{i}"].mean() for i in xaxis])
        names.append(name)

    fig, ax = plt.subplots()
    for i in range(len(data_pts)):
        ax.plot(xaxis, data_pts[i], label=names[i])

    ax.set(xlabel="N Queries", ylabel="Avg time per query")
    ax.legend()
    fig.savefig(save_path, dpi=300)


if __name__ == "__main__":
    MAXX = 25

    COMP_1 = "./runs/timing/date-and-date/path/to/checkpoint.pkl"
    COMP_2 = "./runs/timing/date-and-date/path/to/checkpoint.pkl"
    COMP_3 = "./runs/timing/date-and-date/path/to/checkpoint.pkl"
    DATA = "./timing/data/wikiqa-edited.json"

    comp1 = load(COMP_1)
    comp2 = load(COMP_2)
    comp3 = load(COMP_3)
    data = load(DATA, load_json=True)

    comp1 = process(MAXX, comp1)
    comp2 = process(MAXX, comp2)
    comp3 = process(MAXX, comp3)

    evalutations = {
        "Comp 1": comp1,
        "Comp 2": comp2,
        "Comp 3": comp3,

    }

    plot(MAXX, evalutations)
