# Video Anomaly Detection with SAM 2 and β-CLIP

This project keeps the structured **objects + actions + place + where** prompt learner and β-CLIP classifier, and changes each training sample from one image to a clip sampled from a video-frame folder. SAM 2 segments each frame; β-CLIP assigns object names and produces visual/text embeddings for the proposals.

Important: SAM 2 is a promptable segmentation model, not a language model. SAM 2 produces masks, not object names or text embeddings. Here its automatic masks are matched against an explicit object-name vocabulary with β-CLIP's image/text encoders. The object names and similarity scores are therefore β-CLIP predictions, not native SAM 2 labels.

## Workflow

1. Store each video as a directory of image frames. Use zero-padded filenames such as `000001.jpg` so lexical ordering is temporal ordering.
2. List video directories and video-level class IDs in the CSV manifest.
3. Run the SAM 2 extraction command. It saves per-frame binary masks, object-name metadata, and B-CLIP proposal caches.
4. Train the unchanged structured-prompt classifier on sampled video frames, with cached SAM 2 object proposals as additional conditioning input and object-loss supervision.

SAM 2 segmentation is done frame by frame with its automatic mask generator. The current pipeline does not claim temporal object tracking across frames.

## Install

Install this project and β-CLIP as before:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
bash scripts/setup_bclip.sh
```

Install the official Meta SAM 2 package and its dependencies using its upstream instructions: https://github.com/facebookresearch/sam2. Download a SAM 2 checkpoint and note the matching model YAML. Set `bclip.repo`, `bclip.checkpoint`, `sam2.model_cfg`, and `sam2.checkpoint` in `configs/default.yaml`.

## Dataset Format

`train.csv` and `val.csv` have one row per video. Paths are relative to `data.root` unless absolute:

```csv
video,label
videos/train/normal_0001,0
videos/train/theft_0001,1
videos/train/fire_0001,2
```

Each listed path must be a directory containing image frames. The example categories are normal activity, theft/robbery, and fire. Replace `classes.json` and the object vocabulary to match the dataset being used. Class IDs must be contiguous integers starting at zero.

ShanghaiTech is commonly evaluated as normal/anomaly rather than theft/fire categories. For a binary ShanghaiTech setup, use exactly two video classes (`normal`, `anomaly`), set every anomaly object such as `thief`, `robber`, `fire`, and `smoke` to `class_id: 1` in `sam2.object_vocabulary`, and map normal-scene object names to `class_id: 0`. Do not use the three-class example labels unchanged for a binary dataset.

The existing `anchors.npy` remains the class-level initializer for the structured prompt learner, shaped `[num_classes, anchor_dim]`. SAM 2-derived per-video object visual anchors are stored in the proposal cache and passed dynamically through the B-CLIP model. Random anchors are suitable only for a smoke test:

```bash
python scripts/make_dummy_anchors.py --classes data/classes.example.json --dim 512 --output data/anchors.example.npy
```

## Object Prompts and SAM 2 Extraction

`sam2.object_vocabulary` is distinct from the video classification classes. Each entry has an object `name` and the video `class_id` used as its object-loss target. For example, `thief` and `robber` can remain separate detected text outputs while both map to the theft/robbery class. The default prompt ensemble is:

```text
a CCTV video frame containing {object}
a surveillance camera view of {object}
a video frame showing {object} during an incident
```

Keep vocabulary entries short and concrete (`thief`, `robber`, `fire`, `smoke`, `stolen bag`). Add domain-appropriate terms for each dataset; do not put full action sentences in the object vocabulary. The structured class fields (`object`, `action`, `place`, `where`) remain natural-language descriptors for the existing prompt learner.

Set the matching SAM 2 config/checkpoint paths and run:

```bash
python -m bclip_prompt.extract_sam2 --config configs/default.yaml
```

The command processes videos from both CSV files and writes `data/sam2_cache/<video-key>.npz`, binary mask PNGs under `data/sam2_masks/<video-key>/`, and an `objects.json` per video containing detected object names, mapped class IDs, mask filenames, and β-CLIP similarity. A proposal cache stores the visual region embedding, matched object text embedding, mapped class ID, and validity mask for each frame/proposal.

If changing to this branch from an older cache, rerun extraction. The new cache also stores five normalized mask geometry values per proposal: center x/y, box width/height, and mask area ratio.

## Train and Evaluate

```bash
python -m bclip_prompt.train --config configs/default.yaml
python -m bclip_prompt.evaluate --config configs/default.yaml --checkpoint outputs/best.pt --predictions outputs/video_predictions.csv --frame-scores outputs/frame_anomaly_scores.csv --all-frames
```

`data.frames_per_video` controls temporal sampling. Training and evaluation require the corresponding SAM 2 caches by default; set `data.require_sam2_cache: false` only for a video-classification baseline without object proposals. β-CLIP remains frozen by default, and the original `train_conditioner` option still controls its text-conditioned pooling block. Set `model.anomaly_class_ids` to the class IDs treated as anomalous; for binary normal/anomaly datasets this is usually `[1]`.

## Model and Losses

The model has two video-level branches:

1. **Structured-prompt β-CLIP branch:** each frame is scored against the existing structured class text features. In conditioned mode, the matched SAM 2 object text embeddings and masked-region visual anchors are additional patch-pooling queries. The per-frame class logits are averaged over the sampled frames.
2. **Weakly supervised attention branch:** takes β-CLIP frame embeddings, masked-region visual embeddings, matched object text embeddings, and mask geometry. An object MLP fuses a frame feature and each valid object's visual/text/geometry features. Learned object attention weights pool proposals into a frame representation; learned temporal attention weights pool frame representations into one video representation. A classifier maps this to video-class logits. Invalid padded objects receive zero attention.

For frame `t` and object `k`, let `u[t,k]` be its fused feature and `g[t,k]` indicate whether the proposal is valid. The branch computes:

```text
object_attention[t,:] = masked_softmax(score_object(u[t,:]), g[t,:])
object_context[t] = sum_k object_attention[t,k] * u[t,k]
frame_feature[t] = MLP(frame_embedding[t], object_context[t])
frame_attention[:] = softmax(score_frame(frame_feature[:]))
video_feature = sum_t frame_attention[t] * frame_feature[t]
video_branch_logits = classifier(video_feature)
```

This is hierarchical multiple-instance attention: the video label supervises which frames and objects are useful, without requiring frame-level anomaly labels. It pools across sampled frames but does not explicitly model frame order or track object identities over time. The model exposes `frame_attention` and `object_attention` for inspection. These are attention weights and must not be mistaken for anomaly probabilities.

### Per-Frame Anomaly Scores

The attention branch has a separate anomaly head. For each frame representation `h[t]`, it compares that frame with the mean representation of its video. The anomaly head receives the frame feature and its absolute temporal deviation:

```text
deviation[t] = abs(h[t] - mean_t(h[t]))
frame_anomaly_logit[t] = AnomalyMLP([h[t], deviation[t]])
frame_anomaly_score[t] = sigmoid(frame_anomaly_logit[t])
video_anomaly_logit = mean(top_k(frame_anomaly_logit))
video_anomaly_score = sigmoid(video_anomaly_logit)
```

`model.anomaly_topk` controls top-k pooling; its default of `1` uses the strongest frame as the video-level anomaly evidence. Set it larger when a clip is expected to contain several anomalous frames. The score is a learned weakly supervised ranking/probability-like value, not a ground-truth frame label or guaranteed calibrated probability. Without `--all-frames`, only sampled evaluation frames receive scores. With `--all-frames`, every frame receives a score; `data.inference_frame_chunk_size` controls how many frames pass through β-CLIP at once, while attention and top-k pooling still see the full video.

Video labels become binary anomaly targets through `model.anomaly_class_ids`. The anomaly loss applies binary cross-entropy to the pooled video logit. Since only positive videos are known to contain an anomaly somewhere, a small sparsity penalty on positive clips encourages the frame head to concentrate evidence rather than marking every frame:

```text
L_anomaly = BCEWithLogits(video_anomaly_logit, video_is_anomalous)
					+ λ_sparse * mean(frame_anomaly_score)  # positive videos only
```

The complete objective is:

```text
L = CE(fused_logits, video_class)
	+ λ_branch CE(video_branch_logits, video_class)
	+ λ_anomaly L_anomaly
	+ λ_anchor L_anchor
	+ λ_object L_object
```

Configure `train.anomaly_loss_weight` and `train.anomaly_sparsity_weight`. There is no frame-level anomaly supervision in this objective.

The final class prediction combines normalized evidence from both branches:

```text
fused_logits = log_softmax(bclip_video_logits)
						 + video_branch_weight * log_softmax(video_branch_logits)
```

This avoids adding raw logits with very different scales. `model.video_branch_weight` controls the attention branch's contribution to fused predictions.

The object loss uses the SAM 2/β-CLIP-assigned `class_id` to align each visual region embedding with the corresponding structured β-CLIP class text feature:

```text
L_object = CE(scale * normalize(region_feature) @ normalize(class_text_features).T, object_class_id)
L = CE(fused_logits, video_label)
	+ λ_branch CE(video_branch_logits, video_label)
	+ λ_anchor L_anchor
	+ λ_object L_object
```

Both classification terms use only the video label from the CSV; no frame-level labels are required. `L_object` uses the cached vocabulary-to-class mapping, which is a β-CLIP pseudo-label rather than an independent SAM 2 label. `L_anchor` is the original object-prompt residual consistency regularizer. Configure the terms with `train.video_branch_loss_weight`, `train.object_loss_weight`, and `train.anchor_loss_weight`.

### Training and Evaluation Steps

For each training batch, the loader samples `T` frames per video and retrieves the matching cached object proposals. Training samples temporal positions randomly, but frame resize/center-crop is deterministic so cached regions and mask geometry remain aligned with the frame features. The model computes per-frame β-CLIP scores, class-attention branch scores, frame anomaly logits, top-k video anomaly logits, and fused class scores. It calculates the five loss terms above using the video label and valid proposal targets, backpropagates the weighted sum, and updates the prompt learner and attention branch. β-CLIP remains frozen unless its conditioner is explicitly enabled for training. Checkpoint selection uses fused validation accuracy.

Evaluation runs without gradients and reports fused class accuracy, attention-branch class accuracy, binary anomaly accuracy, and weighted validation loss. It writes one row per video to `outputs/video_predictions.csv` and one row per evaluated frame to `outputs/frame_anomaly_scores.csv` by default. The frame file includes the original frame index/name, video anomaly score, frame anomaly score, and a thresholded flag. Use `--all-frames` for dense scores; otherwise it scores the configured temporal sample. `--anomaly-threshold` controls the display flag; it does not change model training or provide a frame-level accuracy metric without frame labels.

## Notes

- The SAM 2 and β-CLIP checkpoints/repositories are external dependencies and are not included here.
- The object vocabulary-to-class mapping is part of the experiment definition; verify it matches the numeric labels in both CSV files.
- The code expects the released β-CLIP CLIP-compatible text modules and, in conditioned mode, `encode_image_by_block` and `text_conditioned_patches_block`.
