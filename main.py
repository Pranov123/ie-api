import os
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


@app.post("/")
@app.post("/rank")
async def rank(req: Request):
    body = await req.json()

    query = body["query"]
    candidates = body["candidates"]

    texts = [query] + candidates

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )

    embeddings = np.array(
        [item.embedding for item in response.data],
        dtype=np.float32,
    )

    query_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]

    # Normalize embeddings
    query_embedding /= np.linalg.norm(query_embedding)
    candidate_embeddings /= np.linalg.norm(
        candidate_embeddings,
        axis=1,
        keepdims=True,
    )

    # Cosine similarities
    scores = candidate_embeddings @ query_embedding

    # Indices of top 3 candidates
    ranking = np.argsort(scores)[-3:][::-1].tolist()

    return {"ranking": ranking}