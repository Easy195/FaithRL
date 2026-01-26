from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import time
import uvicorn

app = FastAPI()

class PairRequest(BaseModel):
    context: str
    sentences: list[str]
pipe = pipeline("pair-classification", model="your PRM Path", device="cuda:x", trust_remote_code=True)

@app.post("/predict/")
async def predict(pair_request: PairRequest):
    context = pair_request.context
    sentences = pair_request.sentences
    pairs = [(context, sentence) for sentence in sentences]
    sentence_rewards = pipe(pairs)
    
    return {
        "sentence_rewards": sentence_rewards
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8124)
