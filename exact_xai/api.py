from __future__ import annotations

from fastapi import FastAPI
from .schemas import AnswerRequest, AnswerResponse, BatchRequest, BatchResponse
from .pipeline import AnswerPipeline

app = FastAPI(title="EXACT Kaggle-Core xAI API", version="0.1.0")
pipeline = AnswerPipeline(llm=None)

@app.get("/health")
def health():
    return {"status": "ok", "mode": "local_symbolic_api"}

@app.post("/v1/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    return pipeline.answer(req)

@app.post("/v1/batch", response_model=BatchResponse)
def batch(req: BatchRequest):
    return BatchResponse(results=[pipeline.answer(r) for r in req.records])
