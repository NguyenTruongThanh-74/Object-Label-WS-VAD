import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from bclip_prompt.dataset import VideoCSVClassificationDataset, video_cache_key
from bclip_prompt.losses import sam2_object_alignment, weak_anomaly_mil_loss
from bclip_prompt.prompt_learner import StructuredPromptLearner
from bclip_prompt.video_branch import WeaklySupervisedVideoBranch


class FakeTokenizer:
    def __init__(self):
        self.encoder = {"<|startoftext|>": 98, "<|endoftext|>": 99}

    def encode(self, text):
        # deterministic fake word IDs, excluding SOT/EOT
        return [1 + (sum(map(ord, word)) % 90) for word in text.split()]


class FakeAdapter(nn.Module):
    def __init__(self, width=16, context_length=40):
        super().__init__()
        self.text_width = width
        self.context_length = context_length
        self.tokenizer = FakeTokenizer()
        self.embedding = nn.Embedding(100, width)

    @property
    def sot_id(self):
        return 98

    @property
    def eot_id(self):
        return 99

    def token_ids(self, text):
        return self.tokenizer.encode(text)

    def embed_token_ids(self, ids):
        return self.embedding(ids)


def test_prompt_shape_and_anchor_gradient():
    classes = [
        {"id": 0, "object": "ball", "action": "throw", "place": "park", "where": "near tree"},
        {"id": 1, "object": "book", "action": "read", "place": "room", "where": "at desk"},
    ]
    anchors = torch.randn(2, 8)
    adapter = FakeAdapter()
    learner = StructuredPromptLearner(adapter, classes, anchors, {
        "anchor_dim": 8,
        "object_tokens": 1,
        "action_tokens": 1,
        "place_tokens": 1,
        "where_tokens": 1,
        "use_object_residual": True,
    })
    out = learner()
    assert out.embeddings.shape == (2, 40, 16)
    assert out.object_tokens.shape == (2, 1, 16)
    assert torch.all(out.eot_positions > 0)

    out.object_tokens.sum().backward()
    assert learner.anchor_proj.weight.grad is not None
    assert learner.object_residual.grad is not None


def test_sam2_object_alignment_uses_object_labels():
    visual = torch.tensor([[[[1.0, 0.0]]]], requires_grad=True)
    labels = torch.tensor([[[0]]])
    valid = torch.tensor([[[True]]])
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)

    loss = sam2_object_alignment(visual, labels, valid, text, torch.tensor(4.0))
    loss.backward()

    assert loss.item() < 0.1
    assert text.grad is not None


def test_video_dataset_returns_aligned_frames_and_sam2_cache(tmp_path):
    video_dir = tmp_path / "clip_001"
    video_dir.mkdir()
    for frame_index in range(3):
        Image.new("RGB", (20, 20), color=(frame_index * 40, 0, 0)).save(
            video_dir / f"{frame_index:04d}.jpg"
        )
    manifest = tmp_path / "videos.csv"
    manifest.write_text("video,label\nclip_001,1\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    visual = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
    text = visual + 1
    segmentation = np.random.rand(3, 2, 5).astype(np.float32)
    labels = np.ones((3, 2), dtype=np.int64)
    valid = np.ones((3, 2), dtype=np.bool_)
    np.savez_compressed(
        cache_dir / f"{video_cache_key('clip_001')}.npz",
        object_visual_features=visual,
        object_text_features=text,
        object_segmentation_features=segmentation,
        object_labels=labels,
        object_valid=valid,
    )
    dataset = VideoCSVClassificationDataset(
        str(manifest),
        root=str(tmp_path),
        image_size=16,
        frames_per_video=2,
        cache_dir=str(cache_dir),
        max_objects=2,
        object_dim=4,
        train=False,
    )

    frames, label, object_visual, object_text, object_segmentation, object_labels, object_valid, frame_indices = dataset[0]

    assert frames.shape == (2, 3, 16, 16)
    assert label == 1
    assert object_visual.shape == object_text.shape == (2, 2, 4)
    assert object_segmentation.shape == (2, 2, 5)
    assert object_labels.shape == object_valid.shape == (2, 2)
    assert object_valid.all()
    assert frame_indices.tolist() == [0, 2]

    dense_dataset = VideoCSVClassificationDataset(
        str(manifest),
        root=str(tmp_path),
        image_size=16,
        frames_per_video=None,
        cache_dir=str(cache_dir),
        max_objects=2,
        object_dim=4,
        train=False,
    )
    dense_frames, _, _, _, _, _, _, dense_indices = dense_dataset[0]
    assert dense_frames.shape[0] == 3
    assert dense_indices.tolist() == [0, 1, 2]


def test_weak_video_branch_attends_over_objects_and_frames():
    branch = WeaklySupervisedVideoBranch(embed_dim=4, num_classes=3)
    frame_features = torch.randn(2, 3, 4)
    object_visual = torch.randn(2, 3, 2, 4)
    object_text = torch.randn(2, 3, 2, 4)
    segmentation = torch.randn(2, 3, 2, 5)
    valid = torch.tensor([
        [[True, True], [False, False], [True, False]],
        [[True, False], [True, True], [False, False]],
    ])

    output = branch(frame_features, object_visual, object_text, segmentation, valid)
    anomaly_loss = weak_anomaly_mil_loss(
        output["video_anomaly_logit"],
        output["frame_anomaly_logits"],
        torch.tensor([1.0, 0.0]),
    )
    loss = torch.nn.functional.cross_entropy(output["logits"], torch.tensor([0, 2])) + anomaly_loss
    loss.backward()

    assert output["logits"].shape == (2, 3)
    assert output["frame_attention"].shape == (2, 3)
    assert output["object_attention"].shape == (2, 3, 2)
    assert output["frame_anomaly_scores"].shape == (2, 3)
    assert output["video_anomaly_score"].shape == (2,)
    assert torch.allclose(output["frame_attention"].sum(dim=1), torch.ones(2))
    assert torch.allclose(
        output["video_anomaly_logit"], output["frame_anomaly_logits"].max(dim=1).values
    )
    assert torch.all(output["object_attention"][~valid] == 0)
    assert branch.object_fusion[0].weight.grad is not None
    assert branch.anomaly_head[0].weight.grad is not None
