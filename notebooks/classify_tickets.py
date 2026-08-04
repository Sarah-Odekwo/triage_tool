CLASSIFICATION_PROMPT = """You are a support ticket triage assistant for Monzo, a UK digital bank.

Classify the following support ticket. Respond with ONLY a valid JSON object, no other text, no markdown formatting, in this exact structure:

{{
  "category": "Billing" | "Technical" | "Account" | "Other",
  "urgency": "Low" | "Medium" | "High",
  "rationale": "one short sentence explaining your classification"
}}

Category definitions:
- Billing: money movement, charges, fees, refunds, subscriptions, currency conversion
- Technical: app bugs, crashes, card not working due to system issues, login/payment failures
- Account: identity verification, account access/locking, personal detail changes, closing accounts. If a card is blocked and the cause is unclear, default to Account.
- Other: general feedback, feature requests, vague complaints that don't fit above

Urgency definitions:
- High: customer has no access to funds, suspects fraud, or describes a time-critical failure
- Medium: a real problem, not blocking access to money right now
- Low: general questions, feedback, no time pressure

Ticket: "{ticket_text}"
"""

from groq import Groq
from dotenv import load_dotenv
import os
import json
import csv
import time

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def classify_ticket(ticket_text):
    prompt = CLASSIFICATION_PROMPT.format(ticket_text=ticket_text)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2  # low temperature = more consistent classification
    )
    raw_output = response.choices[0].message.content.strip()

    if raw_output.startswith("```"):
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]

    try:
        result = json.loads(raw_output)
        return result
    except json.JSONDecodeError:
        return {"category": "PARSE_ERROR", "urgency": "PARSE_ERROR", "rationale": raw_output}

# Load tickets
rows = []
with open("data/tickets.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Classify each ticket
results = []
for i, row in enumerate(rows):
    print(f"Classifying {i+1}/{len(rows)}: {row['ticket_id']}")
    classification = classify_ticket(row["ticket_text"])
    results.append({
        "ticket_id": row["ticket_id"],
        "ticket_text": row["ticket_text"],
        "true_category": row["true_category"],
        "true_urgency": row["true_urgency"],
        "predicted_category": classification["category"],
        "predicted_urgency": classification["urgency"],
        "rationale": classification["rationale"]
    })
    time.sleep(0.5)  # small delay to avoid rate limits

# Save results
with open("data/classification_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Done. Saved to data/classification_results.csv")