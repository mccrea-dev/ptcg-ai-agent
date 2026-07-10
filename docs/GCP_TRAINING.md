# Training the BC model on Google Cloud

One-time setup is yours (account actions); after that each training run is two
commands.

**Money note (corrected):** the "$3,000 credits for participants" from news
coverage appears to apply to *final-stage* teams, not all entrants — no credit
email exists. Options: GCP's standard $300 new-account trial (plenty for BC
training), or **Kaggle notebooks' free GPU quota (30 h/week)** — zero setup,
data already on Kaggle. Recommendation: use Kaggle's free GPU unless/until we
outgrow it.

## One-time (you), if going the GCP route
1. Create a billing account at console.cloud.google.com (the $300 trial
   activates on signup).
2. Create a project (e.g. `ptcg-abc`), link billing, enable the
   **Compute Engine API**.
3. Request GPU quota if asked: 1× NVIDIA L4 in `us-central1` is plenty
   (Quotas → "GPUs (all regions)" → 1).
4. Install the gcloud CLI locally and run `gcloud auth login` +
   `gcloud config set project ptcg-abc`.

## Per training run
Create the VM (about $0.70/hr; stop it when done):

```
gcloud compute instances create bc-train --zone=us-central1-a \
  --machine-type=g2-standard-4 --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB --maintenance-policy=TERMINATE
```

Ship code + data, train, fetch weights:

```
gcloud compute scp --recurse ptcg pyproject.toml data/train/decisions_v0.jsonl bc-train:~ --zone=us-central1-a
gcloud compute ssh bc-train --zone=us-central1-a --command \
  "pip install numpy && python -m ptcg.training.train_bc_torch decisions_v0.jsonl -o bc_net.pt --epochs 20"
gcloud compute scp bc-train:~/bc_net.pt data/train/ --zone=us-central1-a
gcloud compute instances stop bc-train --zone=us-central1-a
```

(The Deep Learning VM image ships torch+CUDA; our only extra dep is numpy.
`ptcg/engine/sdk` is imported for card data — copy `data/official/sample_submission`
alongside if the run complains, or pre-build the vocab locally.)

## Cost sense
A full 20-epoch run over 54k decisions is minutes on an L4 — pennies. The
credits are really for what comes after: larger replay mining (10k+ episodes),
bigger models, and (stretch) self-play fine-tuning. Don't leave VMs running.
