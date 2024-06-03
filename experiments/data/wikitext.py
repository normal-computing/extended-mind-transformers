import hashlib
import json
import tarfile
from pathlib import Path

import regex as re
import requests

# Download the dataset
fname = "wikitext-103.tar.gz"
url = "https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/" + fname
r = requests.get(url)
Path(fname).write_bytes(r.content)

# Verify the file was downloaded properly by comparing sha512 checksums
sha512sum = "c8186919aa1840af6b734ea41abc580574ea8efe2fafda220f5d01002464d17566d84be5199b875136c9593f0e0678fb5d7c84bb2231de8b4151cb9c83fa2109"
sha512sum_computed = hashlib.sha512(
    Path("wikitext-103.tar.gz").read_bytes()
).hexdigest()
sha512sum == sha512sum_computed

# Extract the dataset
with tarfile.open(fname) as tar:
    tar.extractall()

val_data = Path("wikitext-103/wiki.valid.tokens").read_text()
test_data = Path("wikitext-103/wiki.test.tokens").read_text()

# Store regular expression pattern to search for wikipedia article headings
HEADING_PATTERN = "( \n \n = [^=]*[^=] = \n \n )"

# Split out validation headings and articles
val_split = re.split(HEADING_PATTERN, val_data)
val_headings = [x[7:-7] for x in val_split[1::2]]
val_articles = [x for x in val_split[2::2]]
# Split out test headings and articles
test_split = re.split(HEADING_PATTERN, test_data)
test_headings = [x[7:-7] for x in test_split[1::2]]
test_articles = [x for x in test_split[2::2]]

headings = val_headings + test_headings
inputs = val_articles + test_articles
dataset = [
    {"idx": idx, "heading": heading, "inputs": input}
    for idx, (heading, input) in enumerate(zip(headings, inputs))
]

FILEPATH = "./wikitext.json"
with open(FILEPATH, "w") as f:
    json.dump(dataset, f)
