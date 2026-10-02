# Video Anomaly Detection with SAM 2 and β-CLIP

This project combines the structured **objects + actions + place + where** prompt learner and β-CLIP with weakly supervised video anomaly detection. It predicts anomaly scores at video, frame, and SAM 2 proposal levels, and writes mask-weighted spatial anomaly maps for interpretation.

Important: SAM 2 is a promptable segmentation model, not a language model. SAM 2 produces masks, not object names or text embeddings. Here its automatic masks are matched against an explicit object-name vocabulary with β-CLIP's image/text encoders. The object names and similarity scores are therefore β-CLIP predictions, not native SAM 2 labels.

## Workflow

1. Store each video as a directory of image frames. Use zero-padded filenames such as `000001.jpg` so lexical ordering is temporal ordering.
2. List video directories and video-level class IDs in the CSV manifest.
3. Run SAM 2 to cache object proposals, masks, and B-CLIP features.
4. Train video-level, frame-level, and proposal-level anomaly scores from video labels.
5. Export scored objects and mask-weighted spatial anomaly maps for review.

SAM 2 segmentation is done frame by frame with its automatic mask generator. The current pipeline does not claim temporal object tracking across frames.

## Install

Use Linux or Ubuntu in WSL. SAM 2's official install notes recommend Linux; GPU installation needs a PyTorch/CUDA combination compatible with the machine. The commands below assume the repository root as the working directory.

```bash
conda create -n bclip-vad python=3.10 -y
conda activate bclip-vad
python -m pip install --upgrade pip
```

Install PyTorch and TorchVision for the machine's CUDA version using the [official PyTorch selector](https://pytorch.org/get-started/locally/). SAM 2 currently requires PyTorch 2.5.1+ and TorchVision 0.20.1+.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
bash scripts/setup_bclip.sh
git clone https://github.com/facebookresearch/sam2.git external/sam2
python -m pip install -e external/sam2
```

Follow the [B-CLIP installation instructions](https://github.com/fzohra/B-CLIP) and place its compatible checkpoint at `external/B-CLIP/model/bclip_checkpoint.pth`, or update `bclip.repo` and `bclip.checkpoint` in `configs/default.yaml`. Download a SAM 2.1 checkpoint from the [official SAM 2 repository](https://github.com/facebookresearch/sam2) and place `sam2.1_hiera_large.pt` in `external/sam2/checkpoints/`. The config uses the matching package-relative Hydra config name `configs/sam2.1/sam2.1_hiera_l.yaml`; do not replace it with an absolute path to the YAML file. Update the checkpoint path if you use a different model size.

Check the environment before downloading datasets:

```bash
python -c "import torch, sam2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('sam2', sam2.__file__)"
python -c "from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator; print('SAM 2 import OK')"
```

## Dataset Format

`train.csv` and `val.csv` have one row per video or prepared clip. Paths are relative to `data.root` unless absolute. Every path must be a directory containing ordered image frames; a row pointing to an individual image or an MP4 is not accepted by the dataset loader.

```csv
video,label
data/frames/train/normal/normal_0001,0
data/frames/train/anomalous/anomaly_0001,1
```

Class IDs must be contiguous integers starting at zero. The default config is binary (`0=normal`, `1=anomalous`) and uses [`data/classes.binary.example.json`](data/classes.binary.example.json). Replace the example manifest paths with real extracted data before training.

Do not assign object vocabulary entries to the anomaly class just because their names sound dangerous. In the binary setup, `class_id: null` means that no class-alignment pseudo-label is assigned to that proposal. The video/frame/object anomaly heads learn from video labels and normal-video suppression, allowing an object to be normal or anomalous depending on its visual and scene context.

## Dataset Acquisition and Preparation

The dataset archives are hosted by their research authors or third-party storage providers. Download them yourself, follow each dataset's terms, and do not commit the archives or extracted frames. Keep the official train/test split intact and split by source video, never by frames from the same video.

### ShanghaiTech Campus

Download the ShanghaiTech Campus anomaly dataset from the [SVIP Lab dataset page](https://svip-lab.github.io/dataset/campus_dataset.html) using its Google Drive or OneDrive link. Extract it under `data/raw/shanghaitech/`. The release includes training/testing frames and pixel-level annotations; preserve the provided split and annotation files.

Create a source CSV with `video,label` rows where each `video` points to one sequence/frame directory. Training sequences are normal; a labeled sequence is anomalous if any frame in it has an anomaly annotation. Never use final-test annotations to create training labels or select checkpoints. If the official training split has no anomalous videos, either use a separate labeled validation source or reserve disjoint sequence groups from the released test set for validation and final testing, and disclose that protocol. If the archive presents a sequence as one flat frame directory, that directory is a valid video sample; use `scripts/prepare_video_data.py` to validate it and write the project manifest.

### UCF-Crime

Download UCF-Crime from the [UCF CRCV project page](https://www.crcv.ucf.edu/projects/real-world/), which links the dataset and its official split/annotation files. Extract under `data/raw/ucf_crime/`. Make train and validation manifests from the official training split; keep the official test split untouched. Collapse the 13 crime categories to label `1` and normal videos to label `0` for the default binary task. The source split files have varied formats, so verify each generated path and label against the released instructions rather than random-splitting videos.

### XD-Violence

Download XD-Violence and its annotations from the [official project page](https://roc-ng.github.io/XD-Violence/). It includes weak video labels, multi-label violent categories, and audio. This implementation is video-only: map normal videos to `0` and videos with any violent label to `1`, retain the official split, and record that audio is not used. Do not interpret its multi-label annotations as pixel or object-instance masks.

### Convert Videos or Validate Frame Folders

Create an input manifest for each split with `video,label`. `video` may point either to a raw video file or to a folder that already contains frames. Relative input paths are resolved from `--source-root`. The script reuses frame folders; raw videos are extracted using FFmpeg. By default it preserves every frame, which takes substantial disk space. `--fps 2` is an optional lower-rate extraction and changes the frame/time resolution.

For raw-video datasets, source manifests are simple CSVs. Replace these illustrative paths with the exact filenames in the downloaded archive and use the released split files to decide which rows belong in each split:

```csv
video,label
Videos/Normal_Videos/example_normal.mp4,0
Videos/Arson/example_anomaly.mp4,1
```

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
python scripts/prepare_video_data.py \
	--input-csv data/raw/ucf_train_source.csv \
	--source-root data/raw/ucf_crime \
	--frames-root data/frames/ucf \
	--output-csv data/train.csv
python scripts/prepare_video_data.py \
	--input-csv data/raw/ucf_val_source.csv \
	--source-root data/raw/ucf_crime \
	--frames-root data/frames/ucf \
	--output-csv data/val.csv
```

For already-extracted ShanghaiTech frames, the same command validates directories and writes the manifest without copying frames. Check a few rows and frame directories before running SAM 2:

```bash
head data/train.csv
find data/frames -type f | head
```

Use the same path spelling in the manifest for extraction, training, and evaluation so the proposal-cache keys match. Update `data.train_csv`, `data.val_csv`, and `data.root` if your manifests are stored elsewhere.

The existing `anchors.npy` remains the class-level initializer for the structured prompt learner, shaped `[num_classes, anchor_dim]`. SAM 2-derived per-video object visual anchors are stored in the proposal cache and passed dynamically through the B-CLIP model. Random anchors are suitable only for a smoke test:

```bash
python scripts/make_dummy_anchors.py --classes data/classes.binary.example.json --dim 512 --output data/anchors.binary.example.npy
```

These random anchors are only for a smoke test. Use justified anchors for reported research results.

## Object Prompts and SAM 2 Extraction

`sam2.object_vocabulary` is distinct from video classes. Each entry has an object `name`; `class_id` is optional and should be `null` when that object should not receive a class-alignment pseudo-label. The default binary vocabulary intentionally does not encode anomaly labels from object identity. The prompt ensemble is:

```text
a CCTV video frame containing {object}
a surveillance camera view of {object}
a video frame showing {object} during an incident
```

Keep vocabulary entries short and concrete (`person`, `vehicle`, `gun`, `fire`, `smoke`). Add dataset-appropriate entities; do not put full action sentences in the object vocabulary. The structured class fields (`object`, `action`, `place`, `where`) remain natural-language descriptors for the prompt learner.

Set the matching SAM 2 config/checkpoint paths and run:

```bash
python -m bclip_prompt.extract_sam2 --config configs/default.yaml
```

The command processes videos from both CSV files and writes `data/sam2_cache/<video-key>.npz`, mask PNGs under `data/sam2_masks/<video-key>/`, and an `objects.json` per video. A proposal cache stores visual/text features, optional class targets, validity, and mask geometry. SAM 2 automatic masks are currently generated frame by frame; the current implementation does not ground text prompts to boxes or track object identities across frames.

If changing to this branch from an older cache, rerun extraction. The new cache also stores five normalized mask geometry values per proposal: center x/y, box width/height, and mask area ratio.

## Train and Evaluate

```bash
python -m bclip_prompt.extract_sam2 --config configs/default.yaml
python -m bclip_prompt.train --config configs/default.yaml
python -m bclip_prompt.evaluate --config configs/default.yaml --checkpoint outputs/best.pt --all-frames
```

`data.frames_per_video` controls training/evaluation sampling. Training and evaluation require SAM 2 caches by default; set `data.require_sam2_cache: false` only for a baseline without proposals. β-CLIP remains frozen by default. Evaluation writes video scores to `outputs/video_predictions.csv`, frame scores to `outputs/frame_anomaly_scores.csv`, object scores to `outputs/object_anomaly_scores.csv`, and noisy-OR mask maps under `outputs/spatial_anomaly_maps/`. These maps localize scored SAM 2 proposals; they are not ground-truth pixel segmentation predictions.

Use `--all-frames` to score every prepared frame. Without it, only sampled frames appear in the frame/object CSVs. `--anomaly-threshold` changes the binary display flag, not the learned scores. For the binary datasets here, `model.anomaly_class_ids` is `[1]`.

## Model and Losses

The model has a structured-prompt β-CLIP classification branch and a hierarchical weakly supervised anomaly branch:

1. **Structured-prompt β-CLIP branch:** each frame is scored against the existing structured class text features. In conditioned mode, the matched SAM 2 object text embeddings and masked-region visual anchors are additional patch-pooling queries. The per-frame class logits are averaged over the sampled frames.
2. **Hierarchical anomaly branch:** an object MLP fuses each frame feature with each proposal's visual/text features and mask geometry. An object head predicts `O[t,j]`; noisy-OR combines valid object scores into object-derived frame evidence. A global frame head covers anomalies not explained by a proposal, and a learned gate blends global and object evidence. A temporal convolution enriches frame tokens; temporal attention pools them for video classification. Invalid padded proposals have zero anomaly score and attention.

For frame `t` and object `k`, let `u[t,k]` be its fused feature and `g[t,k]` indicate whether the proposal is valid. The branch computes:

```text
object_attention[t,:] = masked_softmax(score_object(u[t,:]), g[t,:])
O[t,j] = sigmoid(object_anomaly_head(u[t,j]))
A_obj[t] = 1 - product_j(1 - O[t,j])
A_global[t] = sigmoid(global_frame_head(h[t]))
A[t] = gate[t] * A_obj[t] + (1 - gate[t]) * A_global[t]
video_anomaly = mean(top_k(logit(A[t])))
```

The video label supervises top-k frame evidence. Normal videos additionally suppress frame and valid-proposal anomaly scores. Positive frame/object labels are not fabricated. `frame_attention` and `object_attention` are feature-pooling weights, not anomaly probabilities; use the anomaly scores for predictions.

### Per-Frame Anomaly Scores

The global frame head uses each temporally enriched representation and its deviation from the video's mean. Its score is blended with noisy-OR proposal evidence:

```text
deviation[t] = abs(h[t] - mean_t(h[t]))
A_global[t] = sigmoid(GlobalHead([h[t], deviation[t]]))
A_obj[t] = 1 - product_j(1 - O[t,j])
A[t] = gate[t] * A_obj[t] + (1 - gate[t]) * A_global[t]
video_anomaly_score = sigmoid(mean(top_k(logit(A[t]))))
```

`model.anomaly_topk` controls top-k pooling; the default `1` uses the strongest frame as video evidence. Scores are weakly supervised ranking values, not guaranteed calibrated probabilities. Without `--all-frames`, only sampled frames receive scores. With `--all-frames`, every prepared frame is scored; `data.inference_frame_chunk_size` controls β-CLIP memory use.

Video labels become binary anomaly targets through `model.anomaly_class_ids`. The anomaly objective uses top-k video MIL, sparsity on positive clips, normal-frame/object suppression on negative videos, and gated object-to-frame consistency:

```text
L_anomaly = BCEWithLogits(video_anomaly_logit, video_label)
		  + λ_sparse * mean(A[t])                    # positive videos
		  + λ_nf * mean(A[t])                        # normal videos
		  + λ_no * mean(O[t,j])                      # normal videos
		  + λ_h * gate[t] * |A[t] - A_obj[t]|        # valid-object frames
```

The complete objective is:

```text
L = CE(fused_logits, video_class)
	+ λ_branch CE(video_branch_logits, video_class)
	+ λ_anomaly L_anomaly
	+ λ_anchor L_anchor
	+ λ_object L_object
```

Configure `train.anomaly_loss_weight`, `train.anomaly_sparsity_weight`, `train.normal_frame_loss_weight`, `train.normal_object_loss_weight`, and `train.hierarchy_loss_weight`. There are no positive frame/object labels in this objective.

The final class prediction combines normalized evidence from both branches:

```text
fused_logits = log_softmax(bclip_video_logits)
						 + video_branch_weight * log_softmax(video_branch_logits)
```

This avoids adding raw logits with very different scales. `model.video_branch_weight` controls the attention branch's contribution to fused predictions.

The optional object-class alignment loss uses explicitly configured vocabulary `class_id` values. Entries with `class_id: null` are ignored, which is recommended for binary anomaly detection because object identity alone is not an anomaly label:

```text
L_object = CE(scale * normalize(region_feature) @ normalize(class_text_features).T, object_class_id)
L = CE(fused_logits, video_label)
	+ λ_branch CE(video_branch_logits, video_label)
	+ λ_anchor L_anchor
	+ λ_object L_object
```

Both video classification terms use only the video label from the CSV. `L_object` is optional and uses a vocabulary-to-class pseudo-label only when explicitly configured; `L_anchor` regularizes the structured prompt residual. Configure them with `train.video_branch_loss_weight`, `train.object_loss_weight`, and `train.anchor_loss_weight`.

### Training and Evaluation Steps

For each training batch, the loader samples `T` frames per video and retrieves matching cached proposals. Training samples temporal positions randomly, while deterministic image transforms keep frame and proposal features aligned. The model predicts video, frame, and proposal anomaly scores and updates the prompt learner and anomaly branch from video labels. β-CLIP remains frozen unless its conditioner is enabled. Checkpoint selection uses fused validation accuracy.

Evaluation runs without gradients and reports fused class accuracy, anomaly accuracy, and validation loss. It writes video predictions, frame scores, per-object scores, and spatial maps. The spatial map is `1 - product_j(1 - O[t,j] * M[t,j])`; the frame CSV also contains the object-derived frame score. `--all-frames` emits dense outputs; otherwise it scores the configured temporal sample. `--anomaly-threshold` controls display flags only. Frame AUROC and pixel IoU/Dice are not computed by this command.

## Notes

- SAM 2, β-CLIP, and dataset archives are external dependencies and are not included here.
- Current masks are framewise SAM 2 automatic proposals matched to a fixed β-CLIP vocabulary. Text-to-box grounding, object tracking, object-relation transformers, EMA teacher pseudo-labeling, and a learned pixel-refinement decoder are not implemented yet.
- Spatial PNGs are noisy-OR overlays of proposal scores and SAM 2 masks. They localize suspicious proposals, not exact anomalous pixels. Use suitable ground truth and separate metrics before making pixel-segmentation claims.
- ShanghaiTech pixel annotations can support spatial evaluation after correct sequence/frame alignment. UCF-Crime and XD-Violence do not provide dense object-instance masks for this task; spatial results there are qualitative unless an annotated subset is added.
- Keep object vocabulary `class_id` values aligned with video classes only when intentionally enabling the optional region-classification loss; use `null` for ambiguous objects.
- The code expects the released β-CLIP CLIP-compatible text modules and, in conditioned mode, `encode_image_by_block` and `text_conditioned_patches_block`.
