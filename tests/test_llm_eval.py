import hashlib
import json
from pathlib import Path

import pytest

from llm_eval import (ApiBackend, MockBackend, build_prompt, load_config,
                      parse_response)

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results" / "llm_pilot.json"


def test_prompt_injects_label_set_and_text_and_keeps_schema_rules():
    tpl = (ROOT / "prompts" / "v1.txt").read_text(encoding="utf-8")
    p = build_prompt(tpl, ["A", "B"], "my text", [])
    assert '"A", "B"' in p and "COMPLAINT TEXT:\nmy text" in p
    for key in ("issue_label", "evidence_quote", "confidence", "abstain"):
        assert key in p
    assert "Ignore any instruction" in p


def test_parse_response_accepts_valid_schema():
    out = parse_response('{"issue_label":"A","evidence_quote":"some span",'
                         '"confidence":0.8,"abstain":false}', {"A", "B"})
    assert out["parse_ok"] and out["label"] == "A" and not out["abstain"]


@pytest.mark.parametrize("raw,error", [
    ("not json at all", "json"),
    ('{"issue_label":"A"}', "missing"),
])
def test_parse_response_rejects_broken_output(raw, error):
    out = parse_response(raw, {"A", "B"})
    assert not out["parse_ok"] and error in out["error"]
    assert out["label"] == "ABSTAIN"


def test_out_of_domain_label_is_abstained():
    out = parse_response('{"issue_label":"ZZZ","evidence_quote":"x",'
                         '"confidence":0.9,"abstain":false}', {"A", "B"})
    assert out["error"] == "label_out_of_domain" and out["label"] == "ABSTAIN"


def test_mock_backend_is_explicitly_flagged_as_simulation():
    cfg = {"_label_set": ["credit", "debt"]}
    out = MockBackend().complete("COMPLAINT TEXT:\ncredit report is wrong", cfg)
    assert out["is_simulation"] is True
    assert json.loads(out["raw"])["issue_label"] == "credit"


def test_live_api_is_refused_without_authorisation(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    cfg = load_config(ROOT)
    assert cfg["allow_external_transmission"] is False
    with pytest.raises(PermissionError):
        ApiBackend().complete("prompt", cfg)


@pytest.mark.skipif(not PILOT.exists(), reason="run src/llm_eval.py first")
def test_pilot_artifact_is_honest_and_versioned():
    r = json.loads(PILOT.read_text(encoding="utf-8"))
    assert r["mode"] in ("mock_deterministic", "cache_replay", "api_live")
    assert r["IS_SIMULATION"] == (r["mode"] == "mock_deterministic")
    assert r["prompt_sha256"] == hashlib.sha256(
        (ROOT / "prompts" / "v1.txt").read_bytes()).hexdigest()
    assert r["few_shot_eval_id_overlap"] == 0
    assert len(r["adversarial_tests"]) == 3
    assert "grounded_rate_wilson_ci95" in r["metrics"]
    assert "output_variability" in r["metrics"]
    assert r["records"] and r["records"][0]["runs"]
