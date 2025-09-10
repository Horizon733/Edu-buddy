import os, json, time, argparse, asyncio, aiohttp, aiofiles, hashlib
from pathlib import Path
from tqdm.asyncio import tqdm
import re

# ------------------ args ------------------
p = argparse.ArgumentParser()
p.add_argument("--input", "-i", default="data/prompts_for_labeling_v5_en.jsonl", help="JSONL prompts")
p.add_argument("--output", "-o", default="data/train_all_v2.jsonl", help="Merged output JSONL")
p.add_argument("--cache_dir", default="data/.label_cache", help="Cache dir (per-API subdirs auto-added)")
p.add_argument("--concurrency", type=int, default=4, help="parallel requests per API")
p.add_argument("--rate_per_min", type=int, default=40, help="RPM limit per API key")
p.add_argument("--max_labels", type=int, default=100000, help="TOTAL cap across all APIs")
p.add_argument("--temperature", type=float, default=0.4)
p.add_argument("--max_tokens", type=int, default=512)
p.add_argument("--endpoint", default="https://integrate.api.nvidia.com/v1/chat/completions")
p.add_argument("--model", default="openai/gpt-oss-120b")
# NEW: variable number of API keys, space-separated
p.add_argument("--apis", nargs="+", required=True, help="Space-separated API keys: KEY1 KEY2 ...")
args = p.parse_args()

# ------------------ helpers ------------------
BAD_OUTPUT = "Sorry, I could not provide an answer."

LONG_FORM_RX = re.compile(r"\b(compose|symphony|poem|story|essay|article|report|analysis|long\s+explanation)\b", re.I)
CODE_RX      = re.compile(r"\b(code|program|script|abap|python|java|c\+\+|sql|regex|function|class)\b", re.I)

def sha(text): return hashlib.sha256(text.encode("utf-8")).hexdigest()

class RateLimiter:
    def __init__(self, rpm):
        self.capacity = max(1, rpm)
        self.tokens = float(self.capacity)
        self.updated = time.monotonic()
    async def take(self, n=1):
        while True:
            now = time.monotonic()
            refill = (self.capacity/60.0) * (now - self.updated)
            if refill:
                self.tokens = min(self.capacity, self.tokens + refill)
                self.updated = now
            if self.tokens >= n:
                self.tokens -= n
                return
            await asyncio.sleep(0.2)

def build_payload(instr, max_tok, concise_hint=True):
    sysmsg = (
        "You are a knowledgeable, friendly, and safe teacher designed to help students in grades 1 through 12. "
        "Your role is to explain academic topics clearly in English only, using simple, age-appropriate language. "
        "For math or science, show step-by-step reasoning so students learn the process, not just the answer. "
        "Use concise sentences and structured explanations (bullets or numbered steps). "
        "Hard length budget: prefer ≤ 220–300 words. Stop once complete.\n\n"
        "Guidelines:\n"
        "1) English only. 2) Be clear, accurate, concise. 3) Math: steps + units + final answer. "
        "4) Science: short definitions with everyday examples. 5) Reading: short summaries/direct answers. "
        "6) General knowledge: factual, age-appropriate. 7) Current events: brief, neutral. "
        "8) Environmental: clear concepts + practical sustainability examples. "
        "9) If inappropriate/harmful/private or far beyond grade level: refuse politely and redirect."
    )

    user = instr
    # Long-form creative → outline only
    if LONG_FORM_RX.search(instr):
        if concise_hint:
            user = (instr.strip() +
                    "\n\nWrite a concise outline (≤ 200 words). "
                    "If creative output is requested (e.g., symphony/poem/story), provide only: "
                    "• a brief theme/structure • 4–8 bars/motifs in plain text or pseudonotation, not a full piece.")

    # Code-like → allow more tokens, nudge brevity
    effective_max = max_tok
    if CODE_RX.search(instr):
        effective_max = min(max_tok * 3, 1536)
        if concise_hint:
            user = (instr.strip() +
                    "\n\nReturn a single, self-contained code example with brief comments; "
                    "avoid extra prose. If very large, show only the core function.")

    return {
        "model": args.model,
        "messages": [
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": user}
        ],
        "temperature": args.temperature,
        "max_tokens": effective_max
    }

async def fetch_one(session, payload, headers, timeout_s=45, retries=5):
    for attempt in range(1, retries+1):
        try:
            async with session.post(args.endpoint, headers=headers, json=payload, timeout=timeout_s) as r:
                if r.status in (429, 500, 502, 503, 504):
                    await asyncio.sleep(min(2**attempt, 10))
                    continue
                r.raise_for_status()
                return await r.json()
        except Exception:
            if attempt == retries:
                raise
            await asyncio.sleep(min(2**attempt, 10))

def finish_reason_of(res):
    try:
        return (res.get("choices") or [{}])[0].get("finish_reason")
    except Exception:
        return None

def extract_text(res):
    try:
        return (res.get("choices") or [{}])[0].get("message", {}).get("content", "")
    except Exception:
        return ""

async def label_one(session, writer, limiter, instr, cache_dir, headers):
    key = sha(instr)
    cache_fp = cache_dir / f"{key}.json"

    # Reuse cache if valid and non-blank
    if cache_fp.exists():
        try:
            cached = json.loads(cache_fp.read_text(encoding="utf-8"))
            if cached.get("output", "").strip():
                await writer.write(json.dumps(cached, ensure_ascii=False) + "\n")
                return True
            else:
                cache_fp.unlink()
        except:
            pass

    # 1st attempt
    await limiter.take(1)
    payload = build_payload(instr, args.max_tokens, concise_hint=True)
    try:
        res = await fetch_one(session, payload, headers)
        txt = extract_text(res).strip()
        reason = finish_reason_of(res)
    except Exception:
        txt, reason = "", None

    needs_retry = (not txt) or (reason == "length")
    if needs_retry:
        if CODE_RX.search(instr):
            # one higher-budget retry for code
            await limiter.take(1)
            payload2 = build_payload(instr, max_tok=min(args.max_tokens*3, 1536), concise_hint=False)
            try:
                res2 = await fetch_one(session, payload2, headers)
                txt2 = extract_text(res2).strip()
                if txt2:
                    rec = {"instruction": instr, "output": txt2}
                    cache_fp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
                    await writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    return True
            except Exception:
                pass
        else:
            # long-form/creative: reinforce concise outline (no token bump)
            await limiter.take(1)
            payload2 = build_payload(instr, max_tok=args.max_tokens, concise_hint=True)
            try:
                res2 = await fetch_one(session, payload2, headers)
                txt2 = extract_text(res2).strip()
                if txt2:
                    rec = {"instruction": instr, "output": txt2}
                    cache_fp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
                    await writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    return True
            except Exception:
                pass

        # still blank → ensure no cache
        if cache_fp.exists():
            cache_fp.unlink()
        return False

    # good first try
    rec = {"instruction": instr, "output": txt}
    cache_fp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    await writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return True

async def worker(worker_id, prompts, output_fp, cache_dir, api_key, label_cap):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(args.rate_per_min)
    headers = {"Authorization": f"Bearer {api_key}"}
    success = 0
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=args.concurrency)) as session, \
               aiofiles.open(output_fp, "a", encoding="utf-8") as writer:
        tasks = [
            label_one(session, writer, limiter, instr, Path(cache_dir), headers)
            for instr in prompts[:label_cap]
        ]
        desc = f"worker-{worker_id}"
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
            try:
                if await f:
                    success += 1
            except Exception:
                pass
    print(f"✓ {desc} done: {success}")

def split_balanced(lst, n):
    """Split list into n chunks as evenly as possible."""
    k, m = divmod(len(lst), n)
    out = []
    start = 0
    for i in range(n):
        end = start + k + (1 if i < m else 0)
        out.append(lst[start:end])
        start = end
    return out

async def main():
    # load prompts (English-only heuristic)
    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            instr = (o.get("instruction") or o.get("prompt") or "").strip()
            if not instr:
                continue
            if sum(ch.isascii() for ch in instr) < 0.9 * len(instr):
                continue
            rows.append(instr)
    print("total English prompts:", len(rows))

    # cap total across all APIs
    total_cap = min(args.max_labels, len(rows))
    rows = rows[:total_cap]

    # split into N shards, one per API
    N = len(args.apis)
    shards = split_balanced(rows, N)

    # per-API label cap (last shard may carry a remainder automatically)
    base_cap = total_cap // N
    caps = [base_cap] * N
    caps[-1] = total_cap - base_cap * (N - 1)  # remainder to last

    # launch N workers
    tasks = []
    for i, (key, shard) in enumerate(zip(args.apis, shards), start=1):
        out_part = args.output.replace(".jsonl", f"_part{i}.jsonl")
        cache_sub = os.path.join(args.cache_dir, f"api{i}")
        tasks.append(worker(i, shard, out_part, cache_sub, key, caps[i-1]))
    await asyncio.gather(*tasks)

    # merge partials
    out_fp = args.output
    with open(out_fp, "w", encoding="utf-8") as fout:
        for i in range(1, N + 1):
            part = args.output.replace(".jsonl", f"_part{i}.jsonl")
            if os.path.exists(part):
                with open(part, encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
    print(f"✅ merged to {out_fp}")

if __name__ == "__main__":
    asyncio.run(main())
