from groq import Groq
from dotenv import load_dotenv
import os
import json
import csv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

GENERATION_PROMPT = """You are generating realistic synthetic customer support tickets for Monzo, a UK digital bank.

Generate 15 support tickets. Each ticket should read like a real Monzo customer wrote it — natural, sometimes informal, sometimes frustrated, sometimes polite, varying in length (some short, some longer).

Cover a MIX of these categories (roughly even spread across the batch):
- Billing (e.g. incorrect charges, refund requests, subscription issues, currency conversion fees)
- Technical (e.g. app crashes, login issues, card not working, payment failures)
- Account (e.g. account locked, verification issues, changing personal details, closing account)
- Other (e.g. general feedback, feature requests, unclear/vague complaints)

Cover a MIX of urgency levels:
- High (e.g. card blocked with no explanation, suspected fraud, locked out of account with money inside, unable to pay for something urgent)
- Medium (e.g. incorrect charge, app bug affecting usability, unclear billing)
- Low (e.g. general question, minor feature feedback, non-urgent clarification)

Include 2-3 tickets that are deliberately AMBIGUOUS — could reasonably fall into more than one category, or where urgency is unclear from the text alone. These are important edge cases.

Return ONLY a valid JSON array, no other text, in this exact format:
[
  {"ticket_text": "..."},
  {"ticket_text": "..."}
]
"""

all_tickets = []
num_batches = 9  # 9 batches x ~15 tickets ≈ 135 tickets

for i in range(num_batches):
    print(f"Generating batch {i+1}/{num_batches}...")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": GENERATION_PROMPT}],
        temperature=0.9  # higher temperature = more variety between batches
    )
    
    raw_output = response.choices[0].message.content.strip()
    
    # Clean up in case the model wraps output in markdown code fences
    if raw_output.startswith("```"):
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]
    
    try:
        batch = json.loads(raw_output)
        all_tickets.extend(batch)
    except json.JSONDecodeError:
        print(f"Warning: batch {i+1} failed to parse, skipping.")
        continue

# Save to CSV
with open("data/tickets.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ticket_id", "ticket_text", "true_category", "true_urgency"])
    for idx, ticket in enumerate(all_tickets, start=1):
        writer.writerow([f"T{idx:03d}", ticket["ticket_text"], "", ""])

print(f"Done. Generated {len(all_tickets)} tickets, saved to data/tickets.csv")