import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api  # noqa: E402
from plag.features import FEATURE_NAMES, PairFeatures  # noqa: E402
from plag.model import HEAD_FILE  # noqa: E402


class FakeDetector:
    # stands in for the real model so the API tests are fast
    def score(self, source, candidate):
        return 0.9 if source == candidate else 0.1


@pytest.fixture
def client():
    api.app.dependency_overrides[api.get_detector] = lambda: FakeDetector()
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def get_features(source, candidate):
    df = pd.DataFrame({"source_text": [source], "plagiarized_text": [candidate]})
    values = PairFeatures().fit(df).transform(df)[0]
    return dict(zip(FEATURE_NAMES, values))


def test_identical_sentences():
    row = get_features("a man rides a horse", "a man rides a horse")
    assert row["jaccard"] == 1
    assert row["len_ratio_chars"] == 1
    assert row["shared_3gram"] == 3


def test_unrelated_sentences():
    row = get_features("a man rides a horse", "two dogs sleep")
    assert row["jaccard"] == 0
    assert row["shared_2gram"] == 0


def test_api_predict(client):
    same = client.post("/predict", json={"source": "x y", "candidate": "x y"})
    assert same.status_code == 200
    assert same.json()["plagiarised"] is True

    different = client.post("/predict", json={"source": "x y", "candidate": "z"})
    assert different.json()["plagiarised"] is False


def test_api_bad_input(client):
    assert client.post("/predict", json={"source": "", "candidate": "z"}).status_code == 422
    assert client.post("/predict", json={"source": "z"}).status_code == 422


@pytest.mark.skipif(not HEAD_FILE.exists(), reason="trained head not present")
def test_real_model():
    from plag.model import Detector
    detector = Detector()
    source = "A man in a yellow jacket stands on a rock in front of mountains."
    paraphrase = detector.score(source, "A man in a yellow jacket is outdoors.")
    unrelated = detector.score(source, "Two women are cooking dinner indoors.")
    assert paraphrase > 0.5 > unrelated
