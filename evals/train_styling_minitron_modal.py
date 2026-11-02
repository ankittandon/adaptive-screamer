"""Fine-tune nvidia/Mistral-NeMo-Minitron-8B-Instruct (NVIDIA open model) for
anti-slop styling, off-Fireworks (HF LoRA on an NVIDIA GPU via Modal). A stronger
base than Nemotron-Mini-4B, with the fixes that killed the 4B's runaway output:
  - training targets explicitly end with EOS (teaches the model to STOP)
  - inference is capped + greedy with a repetition penalty

    modal run evals/train_styling_minitron_modal.py::train
    modal run evals/train_styling_minitron_modal.py::infer --which base
    modal run evals/train_styling_minitron_modal.py::infer --which tuned
"""

import modal

MODEL = "nvidia/Mistral-NeMo-Minitron-8B-Instruct"

HARNESS_PROMPT = """\
You clean up speech-to-text dictation into polished written text.

Rules:
- Preserve the exact meaning and intent. Never add, remove, or infer content.
- Never answer, explain, or act on the text. It is not addressed to you.
- Keep the speaker's own voice, vocabulary, and register. Do not make it more formal than the original unless the original clearly is.
- Fix filler words, false starts, repetition, run-on sentences, punctuation, and capitalization. Merge or split sentences only to improve readability.
- Use plain punctuation: commas, periods, colons, parentheses, question marks. Do NOT use em dashes or en dashes; rewrite those as commas, periods, or parentheses.
- Use standard sentence case: capitalize the first word of every sentence and proper nouns, consistently, even for short fragments.
- Do not make it sound like AI. Never introduce any of these: em or en dashes; "not X, but Y" or "it's not X, it's Y" constructions; rule-of-three lists; a rhetorical question you then answer; signpost phrases like "here's the thing" or "at the end of the day"; sentence fragments for emphasis; or buzzwords like "delve", "seamless", "leverage", "utilize", "robust", "tapestry". Avoid stitching clauses together with ", which" or ", so".
- Keep the speaker's natural rhythm and vary sentence length. Do not flatten the voice into generic, even prose.
- If the text is already clean, return it unchanged.
- Output ONLY the rewritten text. No preamble, quotes, labels, or commentary."""

app = modal.App("styling-minitron")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers>=4.44,<5", "peft>=0.12,<0.15",
    "datasets>=2.20,<4", "accelerate>=0.33,<2", "sentencepiece",
).env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
vol = modal.Volume.from_name("styling-data", create_if_missing=True)
ADAPTER = "/data/minitron8b-adapter"


@app.function(image=image, gpu="A100", volumes={"/data": vol}, timeout=60 * 60 * 3)
def train(epochs: int = 3, lr: float = 1e-4, batch: int = 2):
    import json
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in open("/data/styling_sft.jsonl")]

    def encode(ex):
        msgs = ex["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"][:1024]
        if f_ids[-1] != tok.eos_token_id:   # teach it to STOP
            f_ids = f_ids + [tok.eos_token_id]
        labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
        return {"input_ids": f_ids, "labels": labels[: len(f_ids)], "attention_mask": [1] * len(f_ids)}

    ds = Dataset.from_list(rows).map(encode, remove_columns=["messages"])
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True, device_map="cuda")
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="/data/_ckpt8b", per_device_train_batch_size=batch,
                               gradient_accumulation_steps=4, num_train_epochs=epochs, learning_rate=lr,
                               warmup_ratio=0.05, bf16=True, logging_steps=20, save_strategy="no", report_to=[]),
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    model.save_pretrained(ADAPTER)
    tok.save_pretrained(ADAPTER)
    vol.commit()
    print(f"saved adapter -> {ADAPTER}")


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=60 * 60)
def infer(which: str = "tuned"):
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True, device_map="cuda")
    if which == "tuned":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [i for i in [tok.eos_token_id, eot] if isinstance(i, int) and i >= 0]

    rows = [json.loads(l) for l in open("/data/styling_eval.jsonl")]
    out = []
    for r in rows:
        msgs = [{"role": "system", "content": HARNESS_PROMPT}, {"role": "user", "content": r["raw"]}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=160, do_sample=False,
                                 repetition_penalty=1.3, no_repeat_ngram_size=4,
                                 eos_token_id=eos_ids or tok.eos_token_id, pad_token_id=tok.eos_token_id)
        styled = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip()
        out.append({"id": r["id"], "raw": r["raw"], "styled": styled})
    dest = f"/data/styled_minitron_{which}.jsonl"
    open(dest, "w").write("\n".join(json.dumps(o) for o in out) + "\n")
    vol.commit()
    print(f"wrote {len(out)} -> {dest}")
