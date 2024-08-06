import json

import pandas as pd
from datasets import load_dataset

EDITS = json.load(open("./experiments/data/edits.json"))
SPLITS = ["2k", "4k", "8k", "16k"]
SAVE_FILEPATH = "./experiments/data/wikiqa-edited.json"


def prepare(split: str):
    dataset = load_dataset("abacusai/WikiQA-Free_Form_QA", split=split)
    df = pd.DataFrame(
        [sample for i, sample in enumerate(dataset) if i % 2 == 0]
    )  # Samples where Question comes first

    df["prompt"] = df["conversations"].apply(
        lambda x: x[0]["value"].split("Document:")[0].strip("\n")
    )
    df["document"] = df["conversations"].apply(
        lambda x: x[0]["value"].split("Document:")[1]
    )
    df["question"] = df["document"].apply(
        lambda x: "Question: " + x.split("Question:")[1].strip("\n")
    )
    df["document"] = df["document"].apply(lambda x: x.split("Question:")[0].strip("\n"))
    df["original_eval"] = df["conversations"].apply(lambda x: x[1]["value"].lower())
    #new evaluation
    df["answer"] = df["original_eval"].apply(
        lambda x: EDITS[x] if x in EDITS else None
    )

    df = df.drop(columns="conversations").drop_duplicates().dropna()

    df["n_replacements"] = df.apply(
        lambda x: x["document"].lower().count(x["original_eval"].lower()), axis=1
    )
    #new document
    df["context"] = df.apply(
        lambda x: "Document: "
        + x["document"].lower().replace(x["original_eval"].lower(), x["answer"]),
        axis=1,
    )
    df["split"] = split
    df.reset_index(drop=True)
    return df.to_dict("records")


if __name__ == "__main__":
    dataset = []
    for split in SPLITS:
        dataset += prepare(split)

    with open(SAVE_FILEPATH, "w") as f:
        json.dump(dataset, f)
