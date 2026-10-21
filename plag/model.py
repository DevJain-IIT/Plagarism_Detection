from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HEAD_FILE = Path(__file__).resolve().parents[1] / "results" / "head.pt"
THRESHOLD = 0.5


def make_head(input_size=384 * 4):
    # small classifier that sits on top of the sentence embeddings
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )


def pair_vector(u, v):
    # u and v are the two sentence embeddings. Besides the embeddings
    # themselves we give the head how far apart they are and how much they agree.
    return np.hstack([u, v, np.abs(u - v), u * v]).astype(np.float32)


class Detector:
    def __init__(self, head_file=HEAD_FILE):
        # one thread is fastest for two short sentences
        torch.set_num_threads(1)
        self.encoder = SentenceTransformer(MODEL_NAME, device="cpu")
        self.head = make_head()
        self.head.load_state_dict(torch.load(head_file, map_location="cpu", weights_only=True))
        self.head.eval()

    def score(self, source, candidate):
        """Probability (0 to 1) that the candidate restates the source."""
        u, v = self.encoder.encode([source, candidate], normalize_embeddings=True,
                                   show_progress_bar=False)
        x = torch.from_numpy(pair_vector(u, v))[None]
        with torch.no_grad():
            return float(torch.sigmoid(self.head(x)).item())
