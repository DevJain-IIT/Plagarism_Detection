import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field

from plag.model import THRESHOLD, Detector


class Pair(BaseModel):
    source: str = Field(min_length=1, max_length=2000)
    candidate: str = Field(min_length=1, max_length=2000)


class Result(BaseModel):
    plagiarised: bool
    score: float
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the model once when the server starts, not on every request
    app.state.detector = Detector()
    app.state.detector.score("warm up", "warm up")
    yield


app = FastAPI(title="Semantic Plagiarism Detection", lifespan=lifespan)


def get_detector(request: Request):
    return request.app.state.detector


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Result)
def predict(pair: Pair, detector=Depends(get_detector)):
    start = time.perf_counter()
    score = detector.score(pair.source, pair.candidate)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return Result(plagiarised=score >= THRESHOLD, score=round(score, 4),
                  latency_ms=round(elapsed_ms, 2))
