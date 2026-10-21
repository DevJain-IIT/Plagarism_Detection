from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HEAD = Path(__file__).resolve().parents[1] / "results" / "head.pt"
THRESHOLD = 0.5


def make_head(dim=384 * 4):
    return torch.nn.Sequential(torch.nn.Linear(dim, 512), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                               torch.nn.Linear(512, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))


def pair_vector(u, v):
    """[u, v, |u-v|, u*v], the same layout the head was trained on."""
    return np.hstack([u, v, np.abs(u - v), u * v]).astype(np.float32)


class Detector:
    def __init__(self, head_path=HEAD):
        torch.set_num_threads(1)  # tiny inputs: threading overhead costs more than it saves
        self.encoder = SentenceTransformer(MODEL, device="cpu")
        self.head = make_head()
        self.head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
        self.head.eval()

    def score(self, source, candidate):
        """Probability that `candidate` restates `source`."""
        u, v = self.encoder.encode([source, candidate], normalize_embeddings=True, show_progress_bar=False)
        with torch.no_grad():
            return float(torch.sigmoid(self.head(torch.from_numpy(pair_vector(u, v))[None])).item())
