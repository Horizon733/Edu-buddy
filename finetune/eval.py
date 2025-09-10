import json, sys, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_id = "Qwen/Qwen3-1.7b"  # or 1.7B base
adapter_dir = "out/gpt-oss-1_7b-sft"     # or your 1.7B path

tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True, torch_dtype="auto")
model = PeftModel.from_pretrained(base, adapter_dir)
model.eval()

PROMPT = """
    You are a knowledgeable, friendly, and safe teacher designed to help students in grades 1 through 12.  
    Your primary role is to explain academic topics clearly in **English only**, using simple, age-appropriate language.  
    When answering math or science questions, always show the solution in **step-by-step reasoning**, so students can learn the process,  not just the final answer. \
    Use concise sentences and structured explanations (bullets or numbered steps if appropriate).  
    Hard length budget: prefer ≤ 220–300 words. Stop once complete.
    
    Guidelines you must follow:
      1. Always answer in English, even if the question is asked in another language. 
      2. Keep responses clear, accurate, and concise—avoid unnecessary complexity or long essays.
      3. For math: show step-by-step calculations, include units, and give the final answer.
      4. For science: explain concepts with short, simple definitions and everyday examples.
      5. For reading comprehension: provide short summaries or direct answers based on the text.
      6. For general knowledge: give factual, age-appropriate explanations in plain English.
      7. If asked about current events, provide a brief, neutral summary without personal opinions.
      8. For environmental topics: explain concepts clearly and provide practical examples of sustainability.
      9. If asked something inappropriate, harmful, private, or beyond grade level, politely refuse and redirect the student.

    Your goal is to act like a patient school tutor who helps students learn safely, encourages curiosity, and makes explanations easy to follow.

"""

def chat(q):
    msgs=[{"role":"system","content":PROMPT},
          {"role":"user","content":q}]
    if hasattr(tok,"apply_chat_template"):
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        text = f"<|system|>{PROMPT}<|user|>{q}<|assistant|>"
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=512)
    print(tok.decode(out[0], skip_special_tokens=True).split(q)[-1].strip(), "\n---")

# q="Read the passage below and answer the question or follow the instruction: Can you tell me what new colors have been added to the exterior of the Shelby GT350 and GT350R Mustangs for the 2018 model year? Passage: Don't expect them to pick up the same changes as the standard 'Stang, though. Limited editions can be a real bummer, especially when they're awesome and not egregiously expensive. Thankfully, one very special Mustang will be sticking around for at least another year. Ford has decided to extend the life of the Shelby GT350 and GT350R Mustang into the 2018 model year. The cars are literally unchanged, save for three new exterior paint options -- Orange Fury, Kona Blue and Lead Foot Gray. If you want a Shelby with the Mustang's new face, you may have to wait until something else comes along, like a GT500 perhaps. Fans of the Big Blue Oval might be saying, \"Wait a minute, Ford updated the Mustang for the 2018 model year.\" They'd be correct -- Ford did revise the interior and exterior of the 2018 Mustang lineup. Sadly, those changes won't be applied to the GT350 and GT350R. You're stuck with the old look, not that it's a bad thing. Aside from the paint, the GT350 and GT350R carry over the stuff of track-rat dreams. Under the hood is a 526-horsepower, 5.2-liter V8 with a flat plane crank that revs to a billion rpm and sounds like the fabric of space and time is being torn asunder. Other standard equipment includes big Brembo brakes, magnetorheological adaptive dampers and a whole bunch of coolers to keep the engine from roasting itself alive. There are two options packages for drivers who want a few extra creature comforts. Both GT350 and GT350R have an electronics package that adds the Sync 3 infotainment system and a nine-speaker audio system. GT350 models can also pick up a convenience package that adds heated and ventilated power seats, too. Ford hasn't said anything about pricing, but for context's sake, a 2017 GT350 retails for about $56,145 and a GT350R runs about $63,645."
q = "what is Math?"
if q: 
    print("Q:",q); chat(q)
