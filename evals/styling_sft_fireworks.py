"""Fine-tune a styling model ON FIREWORKS (supervised LoRA), the way we intended.

Status: this is real, runnable Fireworks code — but on our event account the final
step is blocked: Fireworks returned "dataset uploads are temporarily disabled,
please contact Fireworks AI to get access" on both the direct upload and the
signed-URL path. We kept the code (and ran everything up to the upload) so the
Fireworks fine-tuning path is documented; the actual fine-tunes ran on NVIDIA GPUs
off-Fireworks (see train_*_modal.py) because Fireworks is inference-only for ASR
and the styling SFT upload was blocked.

Base model: nemotron-lightning-3p5-30b-a3b — an NVIDIA open model that is
`supervisedLoraTunable` on Fireworks (training V2). Serving the adapter needs a
dedicated deployment (no serverless LoRA).

    python styling_sft_fireworks.py          # creates dataset, uploads, launches SFT

Data: data/styling_sft.jsonl (319 anti-slop pairs, eval-disjoint). Key: the app
key file or $FIREWORKS_API_KEY.
"""

import json
import os
from pathlib import Path

from fireworks import Fireworks

ACCOUNT = os.environ.get("FIREWORKS_ACCOUNT", "ankit-tandon-mba-6fs")
BASE_MODEL = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
DATASET_ID = "screamer-antislop-v1"
SFT_PATH = "data/styling_sft.jsonl"


def _key():
    k = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if k:
        return k
    return (Path.home() / "Library/Application Support/Screamer/fireworks.key").read_text().strip()


def main():
    c = Fireworks(api_key=_key())
    n = sum(1 for _ in open(SFT_PATH))

    # 1. Create the dataset record.
    try:
        c.datasets.delete(DATASET_ID, account_id=ACCOUNT)
    except Exception:
        pass
    c.datasets.create(
        account_id=ACCOUNT, dataset_id=DATASET_ID,
        dataset={"display_name": DATASET_ID, "example_count": n, "user_uploaded": {}},
    )
    print(f"created dataset {DATASET_ID} ({n} examples)")

    # 2. Upload the JSONL. ** This is where our account is blocked. **
    try:
        with open(SFT_PATH, "rb") as f:
            c.datasets.upload(DATASET_ID, account_id=ACCOUNT, file=f)
        print("uploaded dataset")
    except Exception as e:
        print(f"UPLOAD BLOCKED on this account: {e}")
        print("Request upload access from Fireworks, then re-run — the rest works as-is.")
        return

    # 3. Launch the supervised LoRA fine-tune (training V2).
    job = c.supervised_fine_tuning_jobs.create(
        account_id=ACCOUNT,
        base_model=BASE_MODEL,
        dataset=f"accounts/{ACCOUNT}/datasets/{DATASET_ID}",
        epochs=3,
        learning_rate=1e-4,
        lora_rank=16,
    )
    print(f"launched SFT job: {getattr(job, 'name', job)}")
    print("When done, deploy the adapter to a dedicated deployment and run the head-to-head.")


if __name__ == "__main__":
    main()
