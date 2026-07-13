#!/usr/bin/env python3
"""Judge a saved chat-completion JSON as COHERENT / GARBAGE / UNCERTAIN.

A green /health is not health: a vLLM endpoint can serve numerically-broken garbage
tokens with 200 OK (see docs/ALLIANCE_CLUSTER_GUIDE.md section 2.2). This coherence
canary checks the *content* of a fixed-prompt response for the two garbage flavors
observed on real deployments: multilingual token salad and degenerate repetition
('the the ::'). Conservative by design: only a clear COHERENT verdict should gate
traffic/automation; anything ambiguous is a non-action.
Prints one line: VERDICT reason head=<content head>.

Usage: python3 judge-canary.py response.json   # a /v1/chat/completions response body
The prompt should ask for a small fixed JSON object, e.g.:
  {"action":"explore","reasoning":"<one short sentence>"}
"""
import json
import re
import sys

path = sys.argv[1]
try:
    d = json.load(open(path))
    m = d.get("choices", [{}])[0].get("message", {})
    c = (m.get("content") or "").strip()
except Exception as e:  # error body, truncated file, non-JSON
    print(f"UNCERTAIN parse-error {e!r}")
    raise SystemExit(0)

if len(c) < 10:
    print(f"UNCERTAIN too-short content={c!r}")
    raise SystemExit(0)

nonascii = sum(1 for ch in c if ord(ch) > 127) / len(c)
words = re.findall(r"[a-z]+", c.lower())
the_frac = (words.count("the") / len(words)) if words else 1.0
top_frac = (max(words.count(w) for w in set(words)) / len(words)) if words else 1.0
low = c.lower()
has_json_shape = ('"action"' in low) or ("{" in c and ("explore" in low or "reasoning" in low))

if nonascii > 0.10:
    print(f"GARBAGE multilingual-salad nonascii={nonascii:.2f} head={c[:120]!r}")
elif the_frac > 0.25 or top_frac > 0.40:
    print(f"GARBAGE degenerate-repetition the={the_frac:.2f} top={top_frac:.2f} head={c[:120]!r}")
elif has_json_shape:
    print(f"COHERENT json-shape nonascii={nonascii:.2f} top={top_frac:.2f} head={c[:120]!r}")
else:
    print(f"UNCERTAIN no-json-shape head={c[:200]!r}")
