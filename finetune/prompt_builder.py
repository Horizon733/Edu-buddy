import os, json, random
from datasets import load_dataset

random.seed(7)
os.makedirs("data", exist_ok=True)

# ---- Choose your sources (answers dropped; we only keep prompts) ----
SOURCES = [
    ("garage-bAInd/Open-Platypus", None,"train", 25000),      # reasoning/instructions
    ("databricks/databricks-dolly-15k", None,"train", 15000), # classic instructions
    ("HuggingFaceH4/ultrachat_200k", None,"train_sft", 40000),  # conversational (sampled)
    # ("ai4bharat/samanantar", "hi","train", 30000)            # ✅ Hindi (parallel)
    # ("GEM/xlsum", "hi", 30000),                       # ✅ Hindi (summarization)  -> fallback handled below
]

TARGET_TOTAL = 100_000  # final prompt count (adjust as needed)

def _get(d, *keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def extract_prompt(ex, name):
    """Return a single prompt string or None."""
    # Generic instruction-style fields
    p = _get(ex, "instruction","prompt","question","input","query","problem","text")
    if p: return p

    # UltraChat-style conversations: first user turn
    if "messages" in ex and isinstance(ex["messages"], list):
        for m in ex["messages"]:
            role = str(m.get("role","")).lower()
            if role in ("user","human"):
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()

    # # Samanantar (config 'hi'): Hindi/English parallel → wrap as tasks
    # if name.startswith("ai4bharat/samanantar"):
    #     # common fields seen across dumps
    #     hi = _get(ex, "tgt","hi","sentence_hi") or _get(ex, "src_hi")
    #     en = _get(ex, "src","en","sentence_en") or _get(ex, "src_en")
    #     if hi:
    #         if random.random() < 0.5:
    #             return f"वाक्य पढ़िए और अर्थ सरल हिंदी में समझाइए:\n{hi}"
    #         else:
    #             return f"इस हिंदी वाक्य का अंग्रेज़ी में अनुवाद कीजिए:\n{hi}"
    #     if en:
    #         return f"Translate this to Hindi (हिंदी):\n{en}"

    # # XLSum (Hindi) → summarization prompt
    # if "xlsum" in name:
    #     doc = _get(ex, "document","text")
    #     if doc:
    #         return f"पढ़िए और 2–3 वाक्यों में सारांश लिखिए:\n{doc}"

    # # SQuAD-style fallback
    # if "context" in ex and "question" in ex:
    #     return f"प्रसंग पढ़कर उत्तर दीजिए:\nप्रसंग: {ex['context']}\nप्रश्न: {ex['question']}"

    return None

def load_one(name, split, config, cap):
    """Load up to `cap` prompts from dataset/config (with XLSum fallback)."""
    print(f"loading: {name} {config or ''} (cap={cap})")
    ds = None
    try:
        ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    except Exception as e:
        print("  !! skipping", name, "->", e)
        return []

    rows = []
    # Shuffle if supported, then sample up to cap
    try:
        ds = ds.shuffle(seed=7)
    except Exception:
        pass

    for ex in ds:
        p = extract_prompt(ex, name)
        if p:
            rows.append({"prompt": p})
            if len(rows) >= cap:
                break
    print(f"  -> grabbed {len(rows)}")
    return rows

def add_synthetic(pool):
    # Optional merges of your previous synthetic sets
    for fn in ["data/prompts.jsonl", "data/prompts_hindi_hinglish.jsonl"]:
        if os.path.exists(fn):
            with open(fn, encoding="utf-8") as f:
                for line in f:
                    try:
                        o = json.loads(line)
                        txt = o.get("instruction") or o.get("prompt")
                        if isinstance(txt, str) and txt.strip():
                            pool.append({"prompt": txt.strip()})
                    except:
                        pass
            print(f"added {fn}; total={len(pool)}")
    return pool

def dedup_and_trim(pool, target):
    seen, clean = set(), []
    for r in pool:
        p = r["prompt"].strip()
        if len(p) < 10:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        clean.append({"instruction": p})
        if len(clean) >= target:
            break
    random.shuffle(clean)
    return clean

if __name__ == "__main__":
    pool = []
    for name, cfg, split, cap in SOURCES:
        pool.extend(load_one(name, split, cfg, cap))

    pool = add_synthetic(pool)

    # Light bias toward Hindi (duplicate a random 30% of Devanagari prompts)
    indic_boost = []
    for r in pool:
        if any(ch in r["prompt"] for ch in "अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"):
            if random.random() < 0.30:
                indic_boost.append(r)
    pool.extend(indic_boost)

    clean = dedup_and_trim(pool, TARGET_TOTAL)
    out_path = "data/prompts_for_labeling_v5_en.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for x in clean:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"final prompt count: {len(clean)} -> {out_path}")
