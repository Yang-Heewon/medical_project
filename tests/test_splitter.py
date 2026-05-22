import json
import pandas as pd

from vision_rag_cxr.data.splitters import multilabel_split


def test_splitter_runs():
    rows = []
    for i in range(20):
        labels = {"No Finding": int(i % 2 == 0), "Cardiomegaly": int(i % 3 == 0)}
        rows.append({"uid": str(i), "chexbert_labels_binary": json.dumps(labels)})
    df = pd.DataFrame(rows)
    out = multilabel_split(df, 0.7, seed=0)
    assert set(out["split"]) == {"support", "inference"}
