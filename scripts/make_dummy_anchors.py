import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classes", required=True)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    classes = json.loads(Path(args.classes).read_text())
    rng = np.random.default_rng(args.seed)
    anchors = rng.normal(size=(len(classes), args.dim)).astype("float32")
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True).clip(min=1e-8)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, anchors)
    print(f"saved {anchors.shape} -> {args.output}")


if __name__ == "__main__":
    main()
