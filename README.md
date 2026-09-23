# Structured Learnable Prompting for β-CLIP

Research code for learning a compositional prompt with the fixed semantic order:

**objects + actions + place + and where**

The image branch uses β-CLIP. The text branch injects continuous learnable prompt vectors directly into β-CLIP's CLIP-compatible text transformer.

## Idea

For class `c`, the prompt is assembled as:

```text
objects: [O_c] <object words>
actions: [A] <action words>
place:   [P] <place words>
and where: [W] <where words>
```

where:

- `[O_c] = LN(W_anchor a_c + r_object)` is the object prompt token.
- `a_c` is one externally supplied object anchor vector for class `c`.
- `W_anchor` is a trainable projection into β-CLIP's text-transformer width.
- `r_object` is a learnable residual prompt.
- `[A]`, `[P]`, and `[W]` are shared learnable prompt tokens for action, place, and where.
- The literal descriptor words are embedded by the frozen β-CLIP token embedding, preserving language semantics.

The model can use either:

1. **Global mode**: standard cosine similarity between β-CLIP image features and structured prompt features.
2. **Text-conditioned mode**: β-CLIP patch features are dynamically pooled using each structured prompt as the query before classification.

## Repository layout

```text
bclip_structured_prompt/
├── configs/default.yaml
├── data/
│   ├── classes.example.json
│   └── train.example.csv
├── scripts/
│   ├── setup_bclip.sh
│   └── make_dummy_anchors.py
├── src/bclip_prompt/
│   ├── bclip_adapter.py
│   ├── dataset.py
│   ├── losses.py
│   ├── model.py
│   ├── prompt_learner.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
└── tests/test_prompt.py
```

## 1. Install β-CLIP

The official β-CLIP code is external and is not vendored into this archive.

```bash
bash scripts/setup_bclip.sh
```

Then follow the upstream β-CLIP installation instructions and download either the official β-CLIP checkpoint or the OpenAI CLIP ViT-B/16 checkpoint expected by β-CLIP.

Official project: https://github.com/fzohra/B-CLIP

## 2. Install this project

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## 3. Prepare dataset

`train.csv` / `val.csv` format:

```csv
image,label
images/000001.jpg,0
images/000002.jpg,1
```

`classes.json` format:

```json
[
  {
    "id": 0,
    "name": "person cutting vegetables in kitchen",
    "object": "vegetables knife",
    "action": "cutting",
    "place": "kitchen",
    "where": "on a cutting board",
    "anchor_index": 0
  }
]
```

`anchors.npy` must be `[num_classes, anchor_dim]`, one input anchor vector per class/object prompt.

You can create random anchors only for a smoke test:

```bash
python scripts/make_dummy_anchors.py --classes data/classes.example.json --dim 512 --output data/anchors.example.npy
```

For real research, replace them with meaningful object vectors (for example, object embeddings from a detector, an object encoder, or another frozen semantic encoder).

## 4. Configure β-CLIP

Edit `configs/default.yaml`:

```yaml
bclip:
  repo: external/B-CLIP
  checkpoint: /path/to/bclip_checkpoint.pth
  model_name: CLIP_VITB16_OPENAI
```

The loader uses the official `models_tome.py` factory and supports official β-CLIP checkpoints as well as OpenAI `.pt` checkpoints handled through β-CLIP's conversion functions.

## 5. Train

```bash
python -m bclip_prompt.train --config configs/default.yaml
```

By default β-CLIP is frozen and only the prompt learner / anchor projection are optimized. Set `train.train_conditioner: true` to also fine-tune β-CLIP's text-conditioned pooling block.

## 6. Evaluate

```bash
python -m bclip_prompt.evaluate \
  --config configs/default.yaml \
  --checkpoint outputs/best.pt
```

## Main losses

Classification:

```text
L_cls = CE(logits(image, structured_prompt_c), y)
```

Optional object-anchor regularizer:

```text
L_anchor = 1 - cos(object_prompt_without_residual, object_prompt_with_residual)
```

Final:

```text
L = L_cls + λ_anchor L_anchor
```

## Why the implementation is research-friendly

- Semantic prompt structure is explicit and inspectable.
- Object semantics are anchored by an externally supplied vector rather than learned entirely from scratch.
- Prompt tokens remain differentiable end-to-end.
- Frozen descriptor words preserve natural-language priors.
- β-CLIP can provide query-conditioned image representations instead of a single global image vector.
- The image encoder can remain frozen, making ablations around prompt learning inexpensive.
- The object anchor residual can be disabled to test `anchor only` vs `anchor + learnable residual`.

## Suggested ablations

1. Hard text prompt vs learnable structured prompt.
2. No object anchor vs fixed object anchor vs projected object anchor.
3. Object anchor only vs object anchor + learnable residual.
4. Shared learnable action/place/where tokens vs no learnable context.
5. Global β-CLIP image feature vs text-conditioned patch pooling.
6. Freeze β-CLIP vs fine-tune only the conditioning block.
7. Remove each semantic slot separately.

## Notes

- The code assumes the β-CLIP model exposes the standard CLIP text modules (`token_embedding`, `positional_embedding`, `transformer`, `ln_final`, `text_projection`) and, for conditioned mode, `encode_image_by_block` plus `text_conditioned_patches_block`. These interfaces are present in the released β-CLIP implementation.
- The project deliberately does not copy β-CLIP source or checkpoints into the ZIP. Keep the upstream repository as an external dependency.
