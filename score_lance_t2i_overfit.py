#!/usr/bin/env python3
"""Score generated overfit images against the exact center-cropped target."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--prediction", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def main():
    args = parse_arguments()
    target_path = Path(args.target).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite output: {}".format(output_path))
    target = load_rgb(target_path)
    records = []
    for value in args.prediction:
        path = Path(value).expanduser().resolve()
        prediction = load_rgb(path)
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction {} shape {} differs from target {}".format(
                    path, prediction.shape, target.shape
                )
            )
        difference = prediction - target
        mse = float(np.mean(np.square(difference), dtype=np.float64))
        correlation = float(np.corrcoef(prediction.ravel(), target.ravel())[0, 1])
        records.append(
            {
                "prediction": str(path),
                "mse": mse,
                "rmse": math.sqrt(mse),
                "psnr_db": float("inf") if mse == 0 else -10.0 * math.log10(mse),
                "correlation": correlation,
            }
        )
    aggregate = {
        name: float(np.mean([row[name] for row in records], dtype=np.float64))
        for name in ("mse", "rmse", "psnr_db", "correlation")
    }
    report = {
        "schema_version": 1,
        "target": str(target_path),
        "prediction_count": len(records),
        "average": aggregate,
        "worst": {
            "mse": max(row["mse"] for row in records),
            "rmse": max(row["rmse"] for row in records),
            "psnr_db": min(row["psnr_db"] for row in records),
            "correlation": min(row["correlation"] for row in records),
        },
        "predictions": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
