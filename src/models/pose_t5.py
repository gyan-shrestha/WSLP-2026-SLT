"""Pose-to-text translation: conv frontend + T5 encoder-decoder.

Pose features are projected into T5's embedding space and passed as
`inputs_embeds`, so T5's own encoder does the sequence modelling and its
pretrained decoder does the generation. This is the YouTube-ASL / T5-for-SLT
recipe, and it is deliberately conservative -- see notes/findings.md for why the
novelty budget is spent on register routing and decoding rather than here.

Two additions over the plain recipe:

* **Conv frontend.** 25 fps pose is far more granular than text. Two stride-2
  convolutions downsample 4x before T5 sees anything, which cuts attention cost
  16x and gives each position a ~9-frame receptive field -- closer to the
  timescale of a sign than a single frame is.

* **Register prefix.** A control token prepended in T5's own embedding space.
  The eval set is two disjoint registers identifiable from the uid alone
  (findings F5), so the model is told which one to produce.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Register control tokens. `frag` matches iSign's lowercase fragments, `news`
# the capitalised punctuated prose in the `_clip_` sub-corpus.
REGISTERS = ("frag", "news")
REGISTER_TOKENS = {r: f"<reg_{r}>" for r in REGISTERS}


def _masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool (B, T, D) over valid positions given a (B, T) mask."""
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


class PoseFrontend(nn.Module):
    """(B, T, F) pose features -> (B, T/4, d_model) T5-space embeddings.

    `emb_scale` matters more than it looks. T5 does not scale its embeddings by
    sqrt(d_model), so its learned table has a per-token norm around 300, while a
    LayerNorm'd projection emits norm ~sqrt(d_model) = 27.7. Feeding T5
    `inputs_embeds` at unit scale leaves the pose tokens ~11x smaller than the
    register token we prepend, which swamps them in attention and starts the run
    at a loss of ~600 instead of ~10. The scale is learnable, but initialised to
    the embedding table's actual norm so step 0 is already sane.
    """

    def __init__(self, n_feat: int, d_model: int, dropout: float = 0.1,
                 target_norm: float | None = None, downsample: bool = True):
        super().__init__()
        self.downsample = downsample
        hidden = max(d_model, 512)
        self.proj = nn.Linear(n_feat, hidden)
        # Stride-2 twice: 4x temporal downsample, ~9-frame receptive field.
        # Pose arrives at 25 fps and is far finer-grained than text, so this is
        # right for keypoints. Video features are pre-sampled to 16 frames per
        # clip, and downsampling those to 4 tokens would throw away most of the
        # sequence -- hence stride 1 when `downsample=False`.
        st = 2 if downsample else 1
        self.conv = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=st, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, d_model, kernel_size=5, stride=st, padding=2),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        # LayerNorm output has norm ~sqrt(d_model); rescale to T5's embedding norm.
        init = (target_norm / d_model ** 0.5) if target_norm else 1.0
        self.emb_scale = nn.Parameter(torch.tensor(float(init)))

    def forward(self, feats: torch.Tensor, mask: torch.Tensor):
        x = self.proj(feats)                     # (B, T, H)
        x = self.conv(x.transpose(1, 2))         # (B, d, T/4)
        x = x.transpose(1, 2)                    # (B, T/4, d)
        x = self.drop(self.norm(x)) * self.emb_scale
        # Mask must be downsampled the same way the features were.
        step = 4 if self.downsample else 1
        m = mask[:, ::step][:, : x.shape[1]]
        if m.shape[1] < x.shape[1]:              # conv padding can leave one extra
            m = torch.cat([m, m[:, -1:].expand(-1, x.shape[1] - m.shape[1])], dim=1)
        return x, m


class PoseT5(nn.Module):
    def __init__(self, model_name: str = "google/flan-t5-base", n_feat: int = 206,
                 dropout: float = 0.1, freeze_encoder_layers: int = 0,
                 downsample: bool = True):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

        # Register tokens live in T5's vocabulary so the prefix is an ordinary
        # embedding lookup and the decoder can attend to it like any token.
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(REGISTER_TOKENS.values())}
        )
        self.t5.resize_token_embeddings(len(self.tokenizer))
        self.reg_ids = {
            r: self.tokenizer.convert_tokens_to_ids(t) for r, t in REGISTER_TOKENS.items()
        }

        d_model = self.t5.config.d_model
        # Measured from this checkpoint's table rather than hardcoded -- it
        # differs across T5 variants (~300 for flan-t5-base).
        with torch.no_grad():
            target_norm = self.t5.get_input_embeddings().weight.norm(dim=-1).mean().item()
        self.frontend = PoseFrontend(n_feat, d_model, dropout,
                                     target_norm=target_norm, downsample=downsample)

        if freeze_encoder_layers:
            for blk in self.t5.encoder.block[:freeze_encoder_layers]:
                for p in blk.parameters():
                    p.requires_grad = False

    def _embed(self, feats, mask, registers):
        """Pose embeddings with a register control token prepended."""
        x, m = self.frontend(feats, mask)
        b = x.shape[0]

        ids = torch.tensor(
            [self.reg_ids[r] for r in registers], device=x.device
        ).unsqueeze(1)                                        # (B, 1)
        reg = self.t5.get_input_embeddings()(ids)             # (B, 1, d)

        return (
            torch.cat([reg, x], dim=1),
            torch.cat([torch.ones(b, 1, dtype=m.dtype, device=m.device), m], dim=1),
        )

    def forward(self, feats, mask, registers, labels=None):
        embeds, attn = self._embed(feats, mask, registers)
        return self.t5(inputs_embeds=embeds, attention_mask=attn, labels=labels)

    def contrastive_loss(self, feats, mask, registers, label_ids, temperature=0.07):
        """InfoNCE between pooled pose and pooled target-text encodings.

        Cross-entropy alone lets the decoder coast on iSign's language-model
        prior -- run 1 produced fluent, ungrounded output ("It is a a lot of a
        lot of people") regardless of the pose input. This loss makes the pose
        representation predict *which* sentence it goes with, so the encoder has
        to carry information the decoder cannot invent.

        Both towers share T5's encoder, so this costs one extra encoder pass and
        no extra parameters.
        """
        pose_emb, pose_attn = self._embed(feats, mask, registers)
        pose_h = self.t5.encoder(inputs_embeds=pose_emb,
                                 attention_mask=pose_attn).last_hidden_state
        pose_v = _masked_mean(pose_h, pose_attn)

        text_attn = (label_ids != -100).long()
        text_ids = label_ids.masked_fill(label_ids == -100, self.t5.config.pad_token_id)
        text_h = self.t5.encoder(input_ids=text_ids,
                                 attention_mask=text_attn).last_hidden_state
        text_v = _masked_mean(text_h, text_attn)

        pose_v = nn.functional.normalize(pose_v, dim=-1)
        text_v = nn.functional.normalize(text_v, dim=-1)

        logits = pose_v @ text_v.t() / temperature
        tgt = torch.arange(len(logits), device=logits.device)
        # Symmetric: pose->text and text->pose.
        return 0.5 * (nn.functional.cross_entropy(logits, tgt)
                      + nn.functional.cross_entropy(logits.t(), tgt))

    @torch.no_grad()
    def generate(self, feats, mask, registers, **kw):
        embeds, attn = self._embed(feats, mask, registers)
        # Repetition control is not optional here. With a weak pose signal the
        # decoder falls into loops ("big big big big big big"), which cost chrF
        # outright. Blocking repeated trigrams costs nothing at training time.
        defaults = dict(max_new_tokens=48, num_beams=5, length_penalty=1.0,
                        early_stopping=True, no_repeat_ngram_size=3,
                        repetition_penalty=1.2)
        defaults.update(kw)
        return self.t5.generate(inputs_embeds=embeds, attention_mask=attn, **defaults)

    def decode(self, ids) -> list[str]:
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)
