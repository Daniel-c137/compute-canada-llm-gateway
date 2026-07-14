#!/usr/bin/env python3
"""Canary battery for an OpenAI-compatible chat endpoint: prove it is SANE before traffic.

A green /health is NOT health — a vLLM deployment can serve numerically-broken garbage with
200 OK (docs/ALLIANCE_CLUSTER_GUIDE.md section 2.2). This battery asserts, in order:

  1. smoke        — trivial completion returns 200.
  2. coherence    — a fixed ask-for-JSON prompt; content judged against the two observed
                    garbage flavors (multilingual token salad, degenerate repetition).
  3. toolcall     — native tool-call parse of a single tools=[...] request (tool_choice:auto;
                    never 'required'/named — known FSM hang class on some models).
  4. roundtrip    — turn 2: echo the returned tool-call ID + a tool result back through the
                    chat template. Documented failure classes START at round 2 (sglang#25218).
  5. --warm-ranks — optional concurrent short completions to touch every DP rank.

Greppable verdict lines (stable API for sbatch watchers):
  COHERENCE_VERDICT: COHERENT|GARBAGE|UNCERTAIN ...
  TOOLCALL_CANARY_PASS|TOOLCALL_CANARY_PARTIAL|TOOLCALL_CANARY_FAIL
  TOOLCALL_ROUNDTRIP_PASS|TOOLCALL_ROUNDTRIP_FAIL|TOOLCALL_ROUNDTRIP_SKIP
  CANARY_BATTERY_PASS|CANARY_BATTERY_FAIL

Exit 0 iff coherent AND (tools pass or --no-tools). stdlib only; stream:false throughout
(streaming tool-call parsing is broken upstream for some parsers, e.g. kimi_k2 / vllm#41182).

Usage:
  canary-battery.py --base-url http://HOST:8000/v1 --model org/name \\
      [--thinking-kill-switch] [--no-tools] [--warm-ranks N] [--timeout 600]

Re-run periodically on long-lived serves (mid-campaign sentinel): upstream reports document
coherent-then-garbage degradation appearing only after hours of normal traffic.
"""
import argparse
import concurrent.futures
import json
import re
import sys
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True, help="e.g. http://10.0.0.1:8000/v1")
ap.add_argument("--model", required=True)
ap.add_argument("--thinking-kill-switch", action="store_true",
                help="send chat_template_kwargs:{thinking:false} (thinking-by-default models)")
ap.add_argument("--no-tools", action="store_true", help="skip tool-call phases (non-tool models)")
ap.add_argument("--warm-ranks", type=int, default=0,
                help="fire N concurrent short completions first (touch all DP ranks)")
ap.add_argument("--timeout", type=int, default=600, help="per-request timeout, seconds")
ap.add_argument("--api-key", default="", help="Bearer token if the endpoint requires one")
args = ap.parse_args()

URL = args.base_url.rstrip("/") + "/chat/completions"


def post(body):
    body = dict(body, model=args.model, stream=False)
    if args.thinking_kill_switch:
        body["chat_template_kwargs"] = {"thinking": False}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if args.api_key:
        req.add_header("Authorization", f"Bearer {args.api_key}")
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        return json.load(r)


def msg(resp):
    return resp.get("choices", [{}])[0].get("message", {})


def judge_coherence(content):
    """The two garbage flavors observed on real deployments + a shape check."""
    c = (content or "").strip()
    if len(c) < 10:
        return "UNCERTAIN", f"too-short content={c!r}"
    nonascii = sum(1 for ch in c if ord(ch) > 127) / len(c)
    words = re.findall(r"[a-z]+", c.lower())
    the_frac = (words.count("the") / len(words)) if words else 1.0
    top_frac = (max(words.count(w) for w in set(words)) / len(words)) if words else 1.0
    low = c.lower()
    has_json_shape = ('"action"' in low) or ("{" in c and ("explore" in low or "reasoning" in low))
    if nonascii > 0.10:
        return "GARBAGE", f"multilingual-salad nonascii={nonascii:.2f} head={c[:120]!r}"
    if the_frac > 0.25 or top_frac > 0.40:
        return "GARBAGE", f"degenerate-repetition the={the_frac:.2f} top={top_frac:.2f} head={c[:120]!r}"
    if has_json_shape:
        return "COHERENT", f"json-shape nonascii={nonascii:.2f} top={top_frac:.2f} head={c[:120]!r}"
    return "UNCERTAIN", f"no-json-shape head={c[:200]!r}"


ok = True

# --- 0. optional DP-rank warmup -------------------------------------------------------------
if args.warm_ranks > 0:
    print(f"[warm] firing {args.warm_ranks} concurrent short completions")
    def _one(i):
        try:
            post({"messages": [{"role": "user", "content": f"Say OK ({i})"}], "max_tokens": 8})
            return True
        except Exception as e:
            print(f"[warm] request {i} failed: {e!r}")
            return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.warm_ranks) as ex:
        good = sum(ex.map(_one, range(args.warm_ranks)))
    print(f"[warm] {good}/{args.warm_ranks} ok")

# --- 1. smoke -------------------------------------------------------------------------------
try:
    post({"messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 8})
    print("[smoke] 200 OK")
except Exception as e:
    print(f"CANARY_BATTERY_FAIL (smoke request failed: {e!r})")
    sys.exit(1)

# --- 2. coherence ---------------------------------------------------------------------------
try:
    r = post({"messages": [
        {"role": "system", "content": "You output exactly one JSON object and nothing else."},
        {"role": "user", "content": 'Respond with this JSON only: '
                                    '{"action":"explore","reasoning":"<one short sentence>"}.'},
    ], "max_tokens": 512})
    verdict, why = judge_coherence(msg(r).get("content"))
except Exception as e:
    verdict, why = "UNCERTAIN", f"request-failed {e!r}"
print(f"COHERENCE_VERDICT: {verdict} {why}")
if verdict != "COHERENT":
    ok = False

# --- 3 & 4. tool call + turn-2 round-trip ---------------------------------------------------
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}]
if args.no_tools:
    print("TOOLCALL_ROUNDTRIP_SKIP (--no-tools)")
else:
    turn1_user = {"role": "user",
                  "content": "Look up the current weather in Vancouver using the tool provided."}
    tcs = []
    try:
        r = post({"messages": [turn1_user], "tools": TOOLS, "tool_choice": "auto",
                  "max_tokens": 256})
        m = msg(r)
        tcs = m.get("tool_calls") or []
        fin = r.get("choices", [{}])[0].get("finish_reason")
        print(f"[toolcall] finish_reason={fin} tool_calls={len(tcs)}")
        if tcs:
            f0 = tcs[0].get("function", {})
            print(f"[toolcall] id={tcs[0].get('id')!r} name={f0.get('name')!r} "
                  f"args={f0.get('arguments')!r}")
            if f0.get("name") == "get_weather":
                print("TOOLCALL_CANARY_PASS")
            else:
                print("TOOLCALL_CANARY_PARTIAL (wrong function name)")
                ok = False
        else:
            print(f"TOOLCALL_CANARY_FAIL (no tool_calls; content head: "
                  f"{(m.get('content') or '')[:200]!r})")
            ok = False
    except Exception as e:
        print(f"TOOLCALL_CANARY_FAIL (request failed: {e!r})")
        ok = False

    if not tcs:
        print("TOOLCALL_ROUNDTRIP_SKIP (turn-1 produced no tool_calls)")
    else:
        # IDs echoed VERBATIM — some chat templates re-render them over the whole history and
        # malformed IDs poison later turns.
        fn = tcs[0].get("function", {})
        try:
            r2 = post({"messages": [
                turn1_user,
                {"role": "assistant", "content": msg(r).get("content") or "", "tool_calls": tcs},
                {"role": "tool", "tool_call_id": tcs[0].get("id"), "name": fn.get("name"),
                 "content": '{"city":"Vancouver","temp_c":18,"condition":"cloudy"}'},
            ], "tools": TOOLS, "tool_choice": "auto", "max_tokens": 160})
            m2 = msg(r2)
            c2 = (m2.get("content") or "").strip()
            t2 = m2.get("tool_calls") or []
            print(f"[roundtrip] turn-2 content_len={len(c2)} tool_calls={len(t2)} "
                  f"head={c2[:160]!r}")
            if c2 or t2:
                print("TOOLCALL_ROUNDTRIP_PASS")
            else:
                print("TOOLCALL_ROUNDTRIP_FAIL (empty turn-2 message)")
                ok = False
        except Exception as e:
            print(f"TOOLCALL_ROUNDTRIP_FAIL (turn-2 request failed: {e!r} — "
                  f"template likely rejected the echoed id)")
            ok = False

print("CANARY_BATTERY_PASS" if ok else "CANARY_BATTERY_FAIL")
sys.exit(0 if ok else 1)
