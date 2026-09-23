import torch
import torch.nn as nn

from bclip_prompt.prompt_learner import StructuredPromptLearner


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
