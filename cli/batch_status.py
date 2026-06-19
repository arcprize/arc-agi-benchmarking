"""Check Anthropic batch status. Usage: python cli/batch_status.py <batch_id>"""
import sys
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

if len(sys.argv) < 2:
    print("Usage: python cli/batch_status.py <batch_id>")
    sys.exit(1)

bid = sys.argv[1]
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
b = client.messages.batches.retrieve(bid)
rc = b.request_counts
total = rc.processing + rc.succeeded + rc.errored + rc.canceled + rc.expired
done = rc.succeeded + rc.errored + rc.canceled + rc.expired
pct = done * 100 // total if total else 0

print(f"Batch:      {bid}")
print(f"Status:     {b.processing_status}")
print(f"Created:    {b.created_at}")
print(f"Processing: {rc.processing} | Succeeded: {rc.succeeded} | Errored: {rc.errored} | Canceled: {rc.canceled} | Expired: {rc.expired}")
print(f"Progress:   {done}/{total} ({pct}%)")
