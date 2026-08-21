#!/usr/bin/env python3
"""Ensemble decoding: average logits across checkpoints at every step.

Single models were all we ran until now, which left the most reliable win in
shared tasks on the table. Averaging log-probabilities across independently
seeded models cancels their individual errors -- a token only survives if
several models agree, which is exactly the property a weak-signal task needs.
It also suppresses the degenerate fallback ("This is why he had a lot of
problems in the world"), since models tend to collapse toward *different*
generic sentences.

All members must share a tokenizer and feature preprocessing.

    python src/eval/ensemble.py --ckpts runs/norm/best.pt runs/norm_s1/best.pt \
        --source test -o preds/ensemble.csv
"""

from __future__ import annotations

import argparse
import os
import csv
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import LogitsProcessor, LogitsProcessorList

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import SLTDataset, collate, load_text  # noqa: E402
from eval.metrics import score  # noqa: E402
from models.pose_t5 import PoseT5  # noqa: E402

ROOT = Path(os.environ.get("SLT_ROOT", "."))


def feat_dir_for(cfg: dict, source: str) -> Path:
    """Feature directory for one member, honouring its modality.

    A pose member trained on `feats/isign` must read `feats/test`; a video
    member trained on `vfeats/isign` must read `vfeats/test`. Mixing them
    silently would feed 768-d SigLIP vectors to a model expecting 206-d pose.
    """
    root = Path(cfg["feat_dir"]).parent          # .../feats or .../vfeats
    return root / ("isign" if source == "dev" else source)


def video_of(uid: str) -> str:
    """Video id from a clip uid -- must match train.py so the dev split agrees."""
    m = re.match(r"^(.+?)-+\d+$", uid)
    return m.group(1) if m else uid


class EnsembleLogits(LogitsProcessor):
    """Replace the driving model's logits with the ensemble mean.

    HuggingFace `generate` runs one model; we let the others run in lockstep on
    the same prefix and average their log-probabilities. Averaging in log space
    is a geometric mean over probabilities, which is stricter than an arithmetic
    mean -- a token any member considers unlikely is penalised hard. That is the
    behaviour we want when most members are individually unreliable.
    """

    def __init__(self, others, encs, masks):
        self.others = others
        self.encs = encs
        self.masks = masks

    def __call__(self, input_ids, scores):
        lp = [torch.log_softmax(scores.float(), dim=-1)]
        for m, enc, mask in zip(self.others, self.encs, self.masks):
            out = m.t5(encoder_outputs=enc, attention_mask=mask,
                       decoder_input_ids=input_ids)
            lp.append(torch.log_softmax(out.logits[:, -1, :].float(), dim=-1))
        return torch.stack(lp).mean(0)


@torch.no_grad()
def ensemble_generate(models, batches, device, **gen):
    """Beam search on model 0, per-step logits averaged over all members.

    `batches` is one batch per model, aligned by uid. Members may read
    different modalities (206-d pose vs 768-d SigLIP video), so each embeds its
    own inputs; only the decoder vocabulary is shared.
    """
    regs = batches[0]["registers"]

    embeds, attns, encs = [], [], []
    for m, b in zip(models, batches):
        f, mk = b["feats"].to(device), b["mask"].to(device)
        e, a = m._embed(f, mk, regs)
        embeds.append(e)
        attns.append(a)
        encs.append(m.t5.encoder(inputs_embeds=e, attention_mask=a))

    # Beam search expands the batch to (batch x num_beams) for the driving
    # model, so every other member's encoder state must be expanded the same
    # way or the logits will not align (256 vs 32 at dim 0).
    nb = gen.get("num_beams", 1)
    exp_encs, exp_attns = [], []
    for enc, attn in zip(encs[1:], attns[1:]):
        e = type(enc)(last_hidden_state=enc.last_hidden_state.repeat_interleave(nb, dim=0))
        exp_encs.append(e)
        exp_attns.append(attn.repeat_interleave(nb, dim=0))

    proc = LogitsProcessorList()
    if len(models) > 1:
        proc.append(EnsembleLogits(models[1:], exp_encs, exp_attns))

    return models[0].t5.generate(
        encoder_outputs=encs[0], attention_mask=attns[0],
        logits_processor=proc, **gen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=Path, nargs="+", required=True)
    ap.add_argument("--source", choices=["val", "test", "dev"], default="test",
                    help="'dev' = the labelled iSign held-out split, so the "
                         "ensemble gain can be measured before it is submitted")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=800, help="dev clips to score")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-beams", type=int, default=8)
    ap.add_argument("--length-penalty", type=float, default=2.0)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--min-new-tokens", type=int, default=0,
                    help="forces longer output; took a single model 0.13 -> 0.15 chrF")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models, cfgs, cfg = [], [], None
    for p in a.ckpts:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        cfgs.append(cfg)
        # Members may use different input modalities (pose keypoints vs SigLIP
        # video embeddings), so each one is probed against its own feature dir.
        probe_dir = feat_dir_for(cfg, a.source)
        ds_probe = SLTDataset(probe_dir, None,
                              max_frames=cfg["max_frames"],
                              velocity=cfg.get("velocity", False),
                              clip_value=cfg.get("clip_value", 5.0))
        m = PoseT5(cfg["model_name"], n_feat=ds_probe.n_feat, dropout=0.0)
        m.load_state_dict(ck["model"])
        models.append(m.to(device).eval())
        print(f"loaded {p} (epoch {ck.get('epoch')})", flush=True)

    if a.source == "dev":
        # Same video-level split training used, so this never sees train clips.
        import random as _r
        text = load_text(Path(cfg["text_csv"]), drop_junk=True)
        full = SLTDataset(Path(cfg["feat_dir"]), text, max_frames=cfg["max_frames"],
                          velocity=cfg.get("velocity", False))
        vids = sorted({video_of(u) for u in full.uids})
        rng = _r.Random(cfg["seed"]); rng.shuffle(vids)
        dev_v = set(vids[: max(1, int(len(vids) * cfg["dev_frac"]))])
        uids = [u for u in full.uids if video_of(u) in dev_v][: a.limit]
        ds = SLTDataset(Path(cfg["feat_dir"]), text, cfg["max_frames"], uids=uids,
                        velocity=cfg.get("velocity", False))
    else:
        ds = SLTDataset(ROOT / "feats" / a.source, None, max_frames=cfg["max_frames"],
                        velocity=cfg.get("velocity", False))

    # One loader per member, over that member's own modality. shuffle=False and
    # a shared uid ordering keep the batches aligned row-for-row.
    uid_order = ds.uids
    loaders = []
    for c in cfgs:
        d = SLTDataset(feat_dir_for(c, a.source),
                       ds.text if a.source == "dev" else None,
                       c["max_frames"], uids=uid_order,
                       velocity=c.get("velocity", False),
                       clip_value=c.get("clip_value", 5.0))
        loaders.append(DataLoader(
            d, batch_size=a.batch_size, shuffle=False, num_workers=2,
            collate_fn=lambda b, tk=models[0].tokenizer, ml=c["max_target_len"]:
                collate(b, tk, ml)))
    mods = {Path(c["feat_dir"]).parent.name for c in cfgs}
    print(f"{len(models)}-model ensemble over {len(ds)} clips | modalities: {sorted(mods)}",
          flush=True)

    preds, refs = {}, {}
    for i, bs in enumerate(zip(*loaders)):
        b = bs[0]
        ids = ensemble_generate(models, list(bs), device, num_beams=a.num_beams,
                                length_penalty=a.length_penalty,
                                max_new_tokens=a.max_new_tokens,
                                min_new_tokens=a.min_new_tokens,
                                no_repeat_ngram_size=3, early_stopping=True)
        for uid, txt, ref in zip(b["uids"], models[0].decode(ids), b["texts"]):
            preds[uid] = txt.strip()
            refs[uid] = ref
        if (i + 1) % 20 == 0:
            print(f"  {len(preds)} decoded", flush=True)

    if a.source == "dev":
        order = list(preds)
        sc = score([preds[u] for u in order], [refs[u] for u in order])
        print(f"\nENSEMBLE ({len(models)} models): {sc}", flush=True)
        if not a.out:
            return

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uid", "text"])
        for uid in ds.uids:
            w.writerow([uid, preds.get(uid, " ")])
    lens = [len(v.split()) for v in preds.values()]
    print(f"wrote {a.out} | mean len {sum(lens) / len(lens):.1f} "
          f"| {len(set(preds.values()))} distinct ({len(set(preds.values())) / len(preds):.1%})",
          flush=True)


if __name__ == "__main__":
    main()
