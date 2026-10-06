from fastapi import FastAPI
from pydantic import BaseModel

from mini_agent import run_agent


app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    answer = run_agent(req.message)
    return {"answer": answer}