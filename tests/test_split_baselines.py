import json
from pathlib import Path

import pytest

from baselines import BowNaiveBayes, label_policy, load_folds
from common import SEED
from metrics import macro_f1

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"


def test_bag_of_words_nb_learns_a_simple_signal():
    texts = ["credit report incorrect account status",
             "credit bureau report wrong tradeline",
             "debt collector calls about a debt not owed",
             "collector phone calls for someone else"]
    labels = ["credit", "credit", "debt", "debt"]
    m = BowNaiveBayes(min_df=1).fit(texts, labels)
    assert m.predict(["incorrect credit report tradeline"]) == ["credit"]
    assert m.predict(["collector calls about debt"]) == ["debt"]


def test_rare_classes_are_collapsed_by_label_policy():
    train = [{"issue": "A"}] * 25 + [{"issue": "B"}] * 3
    keep, policy = label_policy(train)
    assert keep == {"A"} and policy["collapsed_records"] == 3
    assert policy["min_support"] == 20


def test_constant_baseline_high_accuracy_low_macro_f1():
    y = ["A"] * 9 + ["B"]
    pred = ["A"] * 10
    assert macro_f1(y, pred) < 0.55
    assert abs(sum(a == b for a, b in zip(y, pred)) / len(y) - 0.9) < 1e-9


@pytest.mark.skipif(not (SPLITS / "split_manifest.json").exists(),
                    reason="run src/split.py first")
def test_frozen_splits_are_disjoint_temporal_and_eval_is_inside_test():
    ids = {k: set((SPLITS / f"{k}_ids.txt").read_text(encoding="utf-8").splitlines())
           for k in ("train", "val", "test")}
    assert not (ids["train"] & ids["val"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["val"] & ids["test"])
    m = json.loads((SPLITS / "split_manifest.json").read_text(encoding="utf-8"))
    assert m["seed"] == SEED
    assert m["cut_date_train_val"] <= m["cut_date_val_test"]
    assert m["folds"]["test"]["date_min"] >= m["cut_date_val_test"]
    ev = set((SPLITS / "llm_eval_ids.txt").read_text(encoding="utf-8").splitlines())
    assert ev <= ids["test"] and len(ev) == m["llm_eval"]["n"]


@pytest.mark.skipif(not (SPLITS / "split_manifest.json").exists(),
                    reason="run src/split.py first")
def test_baseline_label_policy_matches_frozen_training_fold():
    sample = ROOT / "data" / "derived" / "narrative_sample.csv.gz"
    if sample.exists():
        folds = load_folds(ROOT)
        keep, _ = label_policy(folds["train"])
        assert len(keep) > 1
        assert all(r["narrative"] for r in folds["test"])
        return
    # Clean clone: the 40k-row derived plaintext sample is intentionally not
    # versioned (privacy). Check the committed label policy against the frozen
    # split manifest instead, so the same invariant is still verified without
    # redistributing complaint narratives.
    m = json.loads((SPLITS / "split_manifest.json").read_text(encoding="utf-8"))
    res = json.loads((ROOT / "results" / "baseline_metrics.json").read_text(
        encoding="utf-8"))
    pol = res["label_policy"]
    assert pol["min_support"] == 20
    assert pol["kept_classes"] > 1
    assert pol["collapsed_records"] > 0
    assert (pol["kept_classes"] + pol["collapsed_classes"]
            == m["folds"]["train"]["distinct_issues"])
    assert res["n_train"] == m["folds"]["train"]["n"]
    assert res["n_val"] == m["folds"]["val"]["n"]
    assert res["n_test"] == m["folds"]["test"]["n"]
