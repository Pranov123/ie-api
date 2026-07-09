import os
import json
from fastapi import FastAPI, Request
from groq import Groq

app = FastAPI()

client = Groq(api_key=os.environ["GROQ_API_KEY"])


@app.post("/")
@app.post("/rank")
async def rank(req: Request):
    body = await req.json()

    query = body["query"]
    candidates = body["candidates"]

    prompt = f"""
You are an expert semantic retrieval system.

Given a search query and candidate passages,
identify the THREE passages that are most semantically relevant.

Return ONLY valid JSON.

Format:

{{
  "ranking": [i, j, k]
}}

Rules:
- Return exactly 3 integer indices.
- Indices refer to the positions in the candidates array.
- Do NOT explain your reasoning.
- Choose the passages that best answer the query semantically.

Query:

{query}

Candidates:

"""

    for i, c in enumerate(candidates):
        prompt += f"\n[{i}] {c}\n"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a semantic search ranking engine."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = json.loads(response.choices[0].message.content)

    ranking = result.get("ranking", [])

    # Safety cleanup
    ranking = [int(i) for i in ranking if isinstance(i, (int, float))]

    # Ensure exactly 3 indices
    ranking = ranking[:3]

    return {"ranking": ranking}