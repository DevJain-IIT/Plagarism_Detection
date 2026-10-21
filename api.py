import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field

from plag.model import THRESHOLD, Detector


class Pair(BaseModel):
    source: str = Field(min_length=1, max_length=2000)
    candidate: str = Field(min_length=1, max_length=2000)


class Verdict(BaseModel):
    plagiarised: bool
    score: float
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = Detector()  # load once, not per request
    app.state.detector.score("warm up", "warm up")
    yield


app = FastAPI(title="Semantic Plagiarism Detection", lifespan=lifespan)


def get_detector(request: Request):
    return request.app.state.detector


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Verdict)
def predict(pair: Pair, detector=Depends(get_detector)):
    t = time.perf_counter()
    score = detector.score(pair.source, pair.candidate)
    return Verdict(plagiarised=score >= THRESHOLD, score=round(score, 4),
                   latency_ms=round((time.perf_counter() - t) * 1000, 2))
