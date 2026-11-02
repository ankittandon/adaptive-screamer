"""Fine-tune Whisper on TORGO male dysarthric speakers (English) on Modal, loading
the dataset straight from HuggingFace (ungated). Reports base vs fine-tuned WER
overall and stratified by the speaker's first language (accent) — the second
population Screamer adapts to, tying into the non-native-English business case.

    modal run evals/train_l2arctic_modal.py::train

Dataset: resproj007/torgo_dysarthric_male (16 kHz, manual transcripts,
native_language label). Writes per-utterance base/tuned hyps to the volume so
the same bootstrap + McNemar tests apply.
"""

import modal

BASE = "openai/whisper-base.en"
DATASET = "resproj007/torgo_dysarthric_male"

app = modal.App("whisper-torgomale")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers>=4.44,<5",
    "datasets>=2.20,<4",
    "jiwer",
    "soundfile",
    "librosa",
    "accelerate>=0.33,<2",
)
vol = modal.Volume.from_name("torgomale-data", create_if_missing=True)


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=60 * 60 * 3)
def train(epochs: int = 3, lr: float = 1e-5, batch: int = 16, eval_n: int = 0):
    import json
    import re
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    import jiwer
    import torch
    from datasets import load_dataset, Audio
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    proc = WhisperProcessor.from_pretrained(BASE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()

    raw = load_dataset(DATASET)
    train_ds = raw["train"].cast_column("audio", Audio(sampling_rate=16000))
    eval_ds = raw["test"].cast_column("audio", Audio(sampling_rate=16000))
    if eval_n and len(eval_ds) > eval_n:
        eval_ds = eval_ds.select(range(eval_n))
    print(f"train={len(train_ds)} eval={len(eval_ds)}")

    def prep(b):
        a = b["audio"]
        b["input_features"] = proc.feature_extractor(a["array"], sampling_rate=16000).input_features[0]
        b["labels"] = proc.tokenizer(b["text"]).input_ids
        return b

    cols = train_ds.column_names
    tr = train_ds.map(prep, remove_columns=cols)
    ev_refs = [r for r in eval_ds["text"]]
    ev_lang = ["all" for _ in range(len(eval_ds))]
    ev = eval_ds.map(prep, remove_columns=cols)

    @dataclass
    class Collator:
        processor: Any

        def __call__(self, feats: List[Dict[str, Union[List[int], torch.Tensor]]]):
            inp = [{"input_features": f["input_features"]} for f in feats]
            batch = self.processor.feature_extractor.pad(inp, return_tensors="pt")
            lab = self.processor.tokenizer.pad(
                [{"input_ids": f["labels"]} for f in feats], return_tensors="pt"
            )
            labels = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    def transcribe(model):
        model = model.to(device).eval()
        hyps = []
        feats_all = [torch.tensor(f) for f in ev["input_features"]]
        for i in range(0, len(feats_all), 16):
            feats = torch.stack(feats_all[i : i + 16]).to(device)
            with torch.no_grad():
                ids = model.generate(feats, max_new_tokens=225)
            hyps.extend(proc.tokenizer.batch_decode(ids, skip_special_tokens=True))
        return hyps

    def wer_by_lang(hyps, tag):
        buckets, allr, allh = {}, [], []
        rows = []
        for ref, hyp, lang in zip(ev_refs, hyps, ev_lang):
            r, h = _norm(ref), _norm(hyp)
            rows.append({"lang": lang, "ref": ref, tag: hyp})
            if not r:
                continue
            buckets.setdefault(lang, ([], []))
            buckets[lang][0].append(r); buckets[lang][1].append(h)
            allr.append(r); allh.append(h)
        out = {lang: jiwer.wer(rs, hs) * 100 for lang, (rs, hs) in buckets.items()}
        out["ALL"] = jiwer.wer(allr, allh) * 100
        return out, rows

    base_model = WhisperForConditionalGeneration.from_pretrained(BASE)
    base_model.generation_config.max_length = 225
    base_wer, base_rows = wer_by_lang(transcribe(base_model), "base")

    model = WhisperForConditionalGeneration.from_pretrained(BASE)
    model.generation_config.max_length = 225
    args = Seq2SeqTrainingArguments(
        output_dir="/data/whisper-torgomale", per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch, learning_rate=lr, warmup_ratio=0.1,
        num_train_epochs=epochs, fp16=True, eval_strategy="no", save_strategy="no",
        logging_steps=25, report_to=[],
    )
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=tr, data_collator=Collator(proc),
                             processing_class=proc)
    trainer.train()
    tuned_wer, tuned_rows = wer_by_lang(transcribe(model), "tuned")

    # dump per-utterance (ref, base, tuned) for bootstrap + McNemar
    dump = []
    for b, t in zip(base_rows, tuned_rows):
        dump.append({"speaker": b["lang"], "ref": b["ref"], "base": b["base"], "tuned": t["tuned"]})
    open("/data/torgomale_eval_dump.jsonl", "w").write("\n".join(json.dumps(d) for d in dump) + "\n")
    vol.commit()

    print("\n=== TORGO-male base vs fine-tuned WER ===")
    print(f"{'accent':<16}{'base':>9}{'tuned':>9}{'delta':>9}")
    for k in sorted(set(list(base_wer) + list(tuned_wer))):
        b, t = base_wer.get(k, 0), tuned_wer.get(k, 0)
        print(f"{k:<16}{b:>8.1f}%{t:>8.1f}%{b - t:>+8.1f}")
    return {"base": base_wer.get("ALL"), "tuned": tuned_wer.get("ALL")}


@app.local_entrypoint()
def main():
    print(train.remote())
