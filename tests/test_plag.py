import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api  # noqa: E402
from plag.features import NAMES, PairFeatures  # noqa: E402
from plag.model import HEAD  # noqa: E402


class FakeDetector:
    def score(self, source, candidate):
        return 0.9 if source == candidate else 0.1


@pytest.fixture
def client():
    api.app.dependency_overrides[api.get_detector] = lambda: FakeDetector()
    # skip the real model load; the lifespan is only exercised in the model test below
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_features_identical_pair_is_maximal():
    df = pd.DataFrame({"source_text": ["a man rides a horse"], "plagiarized_text": ["a man rides a horse"]})
    f = PairFeatures().fit(df).transform(df)[0]
    row = dict(zip(NAMES, f))
    assert row["jaccard"] == 1 and row["len_ratio_chars"] == 1 and row["shared_3gram"] == 3


def test_features_disjoint_pair_is_zero():
    df = pd.DataFrame({"source_text": ["a man rides a horse"], "plagiarized_text": ["two dogs sleep"]})
    row = dict(zip(NAMES, PairFeatures().fit(df).transform(df)[0]))
    assert row["jaccard"] == 0 and row["shared_2gram"] == 0


def test_api_contract(client):
    r = client.post("/predict", json={"source": "x y", "candidate": "x y"})
    assert r.status_code == 200 and r.json()["plagiarised"] is True
    assert client.post("/predict", json={"source": "x y", "candidate": "z"}).json()["plagiarised"] is False


def test_api_rejects_empty_and_missing(client):
    assert client.post("/predict", json={"source": "", "candidate": "z"}).status_code == 422
    assert client.post("/predict", json={"source": "z"}).status_code == 422


@pytest.mark.skipif(not HEAD.exists(), reason="trained head not present")
def test_real_model_separates_paraphrase_from_unrelated():
    from plag.model import Detector
    d = Detector()
    same = d.score("A man in a yellow jacket stands on a rock in front of mountains.", "A man in a yellow jacket is outdoors.")
    diff = d.score("A man in a yellow jacket stands on a rock in front of mountains.", "Two women are cooking dinner indoors.")
    assert same > 0.5 > diff
