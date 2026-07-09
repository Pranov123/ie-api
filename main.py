import os
import json
from fastapi import FastAPI, Request
from groq import Groq

app = FastAPI()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

REQUIRED_KEYS = [
    "vendor",
    "currency",
    "total_amount",
    "invoice_date",
    "due_in_days",
    "is_paid",
    "priority",
    "contact_email",
    "line_items",
    "item_count",
]


@app.post("/")
@app.post("/invoice-extraction")
async def extract_invoice(req: Request):
    payload = await req.json()

    text = payload.get("text", "")
    schema = payload.get("schema", {})

    prompt = f"""
You are an invoice information extraction engine.

Return ONLY a valid JSON object.

The JSON MUST exactly follow this schema:

{json.dumps(schema, indent=2)}

Extraction rules:

- vendor: exactly as written.
- currency: ISO 4217 code (USD, EUR, GBP, INR, JPY).
- total_amount: integer in the major currency unit.
- invoice_date: YYYY-MM-DD.
- due_in_days: convert payment terms (Net 30 -> 30, two weeks -> 14, etc.).
- is_paid: infer from wording.
- priority: extract or infer ONLY from the document wording. Do NOT derive it from due_in_days.
- contact_email: lowercase.
- line_items: preserve original order.
- item_count: number of line_items.

Return exactly these keys and no others:

vendor
currency
total_amount
invoice_date
due_in_days
is_paid
priority
contact_email
line_items
item_count

Document:

{text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You extract invoice data and return only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
    )

    data = json.loads(response.choices[0].message.content)

    # Normalize fields
    data["currency"] = str(data.get("currency", "")).upper()
    data["contact_email"] = str(data.get("contact_email", "")).lower()
    data["priority"] = str(data.get("priority", "")).lower()

    # Ensure line_items exists
    if not isinstance(data.get("line_items"), list):
        data["line_items"] = []

    normalized_items = []

    for item in data["line_items"]:
        normalized_items.append({
            "sku": str(item.get("sku", "")),
            "quantity": int(item.get("quantity", 0)),
            "unit_price": int(item.get("unit_price", 0)),
        })

    data["line_items"] = normalized_items
    data["item_count"] = len(normalized_items)

    cleaned = {}

    for key in REQUIRED_KEYS:
        cleaned[key] = data.get(key)

    return cleaned