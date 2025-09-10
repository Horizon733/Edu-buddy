import os, json, time, math, argparse, random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, set_seed, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import get_last_checkpoint

# --------------------------- args ---------------------------
p = argparse.ArgumentParser("QLoRA distillation (with ETA)")
p.add_argument("--student", default="Qwen/Qwen3-1.7b", help="HF model id or local path")
p.add_argument("--train_path", default="data/train.jsonl")
p.add_argument("--eval_path", default="data/val.jsonl")
p.add_argument("--out_dir", default="out/qwen3-sft")
p.add_argument("--seq_len", type=int, default=2048)
p.add_argument("--batch_size", type=int, default=1, help="micro-batch per device")
p.add_argument("--grad_accum", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--weight_decay", type=float, default=0.0)
p.add_argument("--warmup_ratio", type=float, default=0.03)
p.add_argument("--num_epochs", type=float, default=1.0)
p.add_argument("--max_steps", type=int, default=-1, help="override epochs if >0")
p.add_argument("--eval_steps", type=int, default=500)
p.add_argument("--save_steps", type=int, default=1000)
p.add_argument("--logging_steps", type=int, default=25)
p.add_argument("--max_train_samples", type=int, default=-1)
p.add_argument("--max_eval_samples", type=int, default=2000)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--bf16", action="store_true", help="use bf16 mixed precision")
p.add_argument("--fp16", action="store_true", help="use fp16 mixed precision")
p.add_argument("--gradient_checkpointing", action="store_true", help="enable grad ckpt")
p.add_argument("--packing", action="store_true", help="pack multiple samples per seq_len")
p.add_argument("--use_chat_template", action="store_true", help="tokenizer.apply_chat_template")
p.add_argument("--lora_r", type=int, default=16)
p.add_argument("--lora_alpha", type=int, default=32)
p.add_argument("--lora_dropout", type=float, default=0.05)
p.add_argument("--lora_target", default="q_proj,k_proj,v_proj,o_proj", help="comma-separated module names")
p.add_argument("--adam8bit", action="store_true", help="use 8-bit optim states via bitsandbytes")
p.add_argument("--wandb", default="", help="WANDB project name (empty = disabled)")
p.add_argument("--save_merged", action="store_true", help="save merged FP16 model at end (requires VRAM/RAM)")
args = p.parse_args()

print("Training config:", json.dumps(vars(args), indent=2), flush=True)

# --------------------------- seed ---------------------------
set_seed(args.seed)

# --------------------------- tokenizer ----------------------
print("[INFO] Loading tokenizer:", args.student, flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    args.student, use_fast=True, trust_remote_code=True
)
# Qwen chat template often needs EOS & padding defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --------------------------- dataset ------------------------
def jsonl_to_hf(path):
    return load_dataset("json", data_files=path, split="train")

def row_to_text(ex: Dict) -> str:
    """
    Expect JSONL rows with keys: instruction, output (teacher answer).
    Skips bad/blank rows.
    """
    instr = (ex.get("instruction") or ex.get("prompt") or "").strip()
    out = (ex.get("output") or "").strip()
    if not instr or not out:
        return ""
    if args.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content":
             "You are a helpful, safe tutor for grades 5–10. Answer in English. "
             "Use concise explanations and step-by-step reasoning for math."},
            {"role": "user", "content": instr},
            {"role": "assistant", "content": out},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # Fallback plain format
    return f"<|system|>\nYou are a helpful, safe tutor for grades 5–10. Answer in English.\n" \
           f"<|user|>\n{instr}\n<|assistant|>\n{out}"

print("[INFO] Loading train set:", args.train_path, flush=True)
train_raw = jsonl_to_hf(args.train_path)
if args.max_train_samples > 0:
    train_raw = train_raw.select(range(min(args.max_train_samples, len(train_raw))))

print("[INFO] Loading eval set:", args.eval_path, flush=True)
eval_raw = jsonl_to_hf(args.eval_path)
if args.max_eval_samples > 0:
    eval_raw = eval_raw.select(range(min(args.max_eval_samples, len(eval_raw))))

def map_fn(batch):
    texts = []
    for ex in batch["instruction"] if "instruction" in batch else [None]:
        pass
    out = []
    for ex in batch:
        t = row_to_text(ex)
        if t: out.append({"text": t})
    return out

train_ds = train_raw.map(lambda ex: {"text": row_to_text(ex)}, remove_columns=train_raw.column_names)
eval_ds  = eval_raw.map(lambda ex: {"text": row_to_text(ex)},  remove_columns=eval_raw.column_names)

# filter empties
train_ds = train_ds.filter(lambda x: len(x["text"]) > 0)
eval_ds  = eval_ds.filter(lambda x: len(x["text"]) > 0)

print(f"[INFO] Train samples: {len(train_ds)} | Eval samples: {len(eval_ds)}", flush=True)

# --------------------------- 4-bit & LoRA -------------------
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16
)

print("[INFO] Loading student in 4-bit:", args.student, flush=True)
model = AutoModelForCausalLM.from_pretrained(
    args.student,
    quantization_config=bnb_cfg,
    torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    device_map="auto",
    trust_remote_code=True
)

if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

target_modules = [t.strip() for t in args.lora_target.split(",") if t.strip()]
print("[INFO] LoRA targets:", target_modules, flush=True)
peft_cfg = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    task_type=TaskType.CAUSAL_LM,
    target_modules=target_modules,
    bias="none"
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

# --------------------------- collator -----------------------
# Packing with TRL SFTTrainer handles chunking; if not packing, standard LM collator works too.
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --------------------------- training args ------------------
total_batch_size = args.batch_size * args.grad_accum  # per device; single-GPU assumed
run_name = os.path.basename(args.out_dir.rstrip("/"))

common_kwargs = dict(
    output_dir=args.out_dir,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    logging_steps=args.logging_steps,
    evaluation_strategy="steps",
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    save_total_limit=3,
    bf16=args.bf16,
    fp16=(args.fp16 and not args.bf16),
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    gradient_checkpointing=args.gradient_checkpointing,
    optim="paged_adamw_8bit" if args.adam8bit else "adamw_torch",
    ddp_find_unused_parameters=False,
    report_to=("wandb" if args.wandb else "none"),
    run_name=(args.wandb or run_name),
)

use_cuda = torch.cuda.is_available()
# prefer fused AdamW if CUDA & supported
if args.adam8bit:
    if use_cuda:
        chosen_optim = "paged_adamw_8bit"
    else:
        print("[WARN] 8-bit optimizer unavailable (no CUDA). Falling back to 32-bit AdamW.")
        chosen_optim = "adamw_torch"
else:
    if use_cuda and torch.cuda.get_device_capability(0)[0] >= 7:
        chosen_optim = "adamw_torch_fused"   # faster on many NVIDIA GPUs
    else:
        chosen_optim = "adamw_torch"

print(f"[INFO] Optimizer: {chosen_optim}")

train_args = SFTConfig(
    output_dir=args.out_dir,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    logging_steps=args.logging_steps,
    eval_strategy="steps",           # TRL uses eval_strategy, not evaluation_strategy
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    save_total_limit=3,
    bf16=args.bf16,
    fp16=(args.fp16 and not args.bf16),
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    gradient_checkpointing=args.gradient_checkpointing,
    optim=chosen_optim,
    run_name=(args.wandb or run_name),
    report_to=("wandb" if args.wandb else "none"),
    max_length=args.seq_len,
    num_train_epochs=0 if args.max_steps > 0 else args.num_epochs,
    max_steps=args.max_steps,
    packing=args.packing,
    dataset_text_field="text"
)

# --------------------------- ETA callback -------------------
class ETACallback(TrainerCallback):
    def __init__(self, print_every=50):
        self.print_every = print_every
        self.start = None
        self.last_ts = None
        self.last_step = 0
        self.smooth = []

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.start = time.time()
        self.last_ts = self.start
        self.last_step = state.global_step or 0
        print(f"[ETA] Training starts. target_steps={state.max_steps or 'epoch-based'}", flush=True)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # average step time over a short window
        now = time.time()
        steps_done = state.global_step
        delta_steps = steps_done - self.last_step
        delta_t = now - self.last_ts
        if delta_steps > 0 and delta_t > 0:
            step_time = delta_t / max(1, delta_steps)
            self.smooth.append(step_time)
            if len(self.smooth) > 50:
                self.smooth.pop(0)
        self.last_ts = now
        self.last_step = steps_done

        if steps_done % self.print_every == 0 and steps_done > 0:
            avg_step_t = sum(self.smooth)/len(self.smooth) if self.smooth else None
            if state.max_steps and avg_step_t:
                remain = state.max_steps - steps_done
                eta_s = remain * avg_step_t
                done_pct = 100.0 * steps_done / max(1, state.max_steps)
                print(f"[ETA] step {steps_done}/{state.max_steps} ({done_pct:.1f}%) | "
                      f"avg_step {avg_step_t:.3f}s | "
                      f"ETA {int(eta_s//3600)}h{int((eta_s%3600)//60)}m{int(eta_s%60)}s",
                      flush=True)
            else:
                elapsed = now - self.start
                print(f"[ETA] step {steps_done} | elapsed {int(elapsed//3600)}h{int((elapsed%3600)//60)}m",
                      flush=True)

eta_cb = ETACallback(print_every=max(25, args.logging_steps))

# --------------------------- info about batches -------------------
num_train_samples = len(train_ds)
tokens_per_sample = args.seq_len
effective_batch = args.batch_size * args.grad_accum

# estimate steps per epoch (samples / effective_batch)
steps_per_epoch = math.ceil(num_train_samples / effective_batch)

print("=" * 60)
print(f"[INFO] Effective batch size : {effective_batch} samples")
print(f"[INFO] Train samples        : {num_train_samples}")
print(f"[INFO] Steps per epoch      : {steps_per_epoch}")
if args.max_steps > 0:
    print(f"[INFO] Max steps (override) : {args.max_steps}")
print("=" * 60, flush=True)

# --------------------------- trainer ------------------------
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds if len(eval_ds) > 0 else None,
    # data_collator=collator
    args=train_args,
)

# --------------------------- GO -----------------------------
os.makedirs(args.out_dir, exist_ok=True)
if args.wandb:
    os.environ["WANDB_PROJECT"] = args.wandb

print("[INFO] Starting training...", flush=True)
trainer.add_callback(eta_cb)
# trainer.train(resume_from_checkpoint=True)
last_ckpt = None
if os.path.isdir(args.out_dir):
    try:
        last_ckpt = get_last_checkpoint(args.out_dir)
    except Exception:
        last_ckpt = None  # e.g., dir exists but isn't a Trainer checkpoint

if last_ckpt:
    print(f"[INFO] Resuming from checkpoint: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
else:
    # If the directory exists and is non-empty but not a valid checkpoint,
    # either clear it or let Trainer overwrite.
    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir):
        print(f"[WARN] Output dir {args.out_dir} exists and has no valid checkpoint. "
              f"Starting fresh. (Files may be overwritten.)")
    trainer.train()

print("[INFO] Training complete. Saving PEFT adapter...", flush=True)
trainer.save_model(args.out_dir)   # saves adapter

# Optionally save merged FP16 model for inference (bigger RAM need)
if args.save_merged:
    print("[INFO] Merging LoRA into base (this may take a while)...", flush=True)
    from peft import merge_and_unload
    full = trainer.model
    full = merge_and_unload(full)
    full.to(torch.float16)
    full.save_pretrained(os.path.join(args.out_dir, "merged_fp16"))
    tokenizer.save_pretrained(os.path.join(args.out_dir, "merged_fp16"))
    print("[INFO] Saved merged model at:", os.path.join(args.out_dir, "merged_fp16"))
# accelerate launch --dynamo_backend no scripts/distill_qLoRA_eta.py --train_path data/train.jsonl --eval_path data/val.jsonl --out_dir out/gpt-oss-1_7b-sft --seq_len 2048 --batch_size 1 --grad_accum 16 --lr 1e-4 --num_epochs 1 --fp16 --gradient_checkpointing --packing --eval_steps 400 --save_steps 800 --logging_steps 25
