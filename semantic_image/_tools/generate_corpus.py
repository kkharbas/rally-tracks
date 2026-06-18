#!/usr/bin/env python3
"""
Generate a Rally corpus from a directory of images.

Each output line is a JSON document matching the semantic_image track mapping:
  {"name": "<stem>", "image": {"type": "image", "value": "data:<mime>;base64,<b64>"}}

Usage:
  python generate_corpus.py <image_dir> [--output <path>] [--format bz2|gz|jsonl]

Output file defaults to <image_dir_name>-documents.json.bz2 in the current directory.

After the run, the script prints the three numbers you need for track.json:
  document-count, compressed-bytes, uncompressed-bytes
"""

import argparse
import base64
import bz2
import gzip
import io
import json
import mimetypes
import os
import sys

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB


def resize_to_limit(path: str, mime_type: str) -> bytes:
    """Return image bytes, resizing proportionally if the file exceeds MAX_IMAGE_BYTES."""
    file_size = os.path.getsize(path)
    if file_size <= MAX_IMAGE_BYTES:
        with open(path, "rb") as fh:
            return fh.read()

    if not _PIL_AVAILABLE:
        raise RuntimeError(
            f"{os.path.basename(path)} is {file_size / 1024 / 1024:.1f} MB > 5 MB "
            "and Pillow is not installed. Run: pip install Pillow"
        )

    fmt = "JPEG" if mime_type == "image/jpeg" else "PNG"
    save_kwargs = {"format": fmt, "optimize": True}
    if fmt == "JPEG":
        save_kwargs["quality"] = 85

    img = Image.open(path)
    # Scale down proportionally until the encoded size fits.
    # Each iteration reduces linear dimensions by ~30% (~50% area).
    scale = 1.0
    while True:
        if scale < 1.0:
            new_w = max(1, int(img.width * scale))
            new_h = max(1, int(img.height * scale))
            resized = img.resize((new_w, new_h), Image.LANCZOS)
        else:
            resized = img

        buf = io.BytesIO()
        resized.save(buf, **save_kwargs)
        data = buf.getvalue()
        if len(data) <= MAX_IMAGE_BYTES:
            return data

        scale *= 0.7  # reduce by ~30% per iteration

        if scale < 0.01:
            raise RuntimeError(f"Cannot shrink {os.path.basename(path)} below 5 MB")


def encode_image(path: str) -> tuple[dict, bool]:
    """Return (document, was_resized)."""
    mime_type, _ = mimetypes.guess_type(path)
    if not mime_type:
        mime_type = "image/jpeg"

    original_size = os.path.getsize(path)
    data = resize_to_limit(path, mime_type)
    was_resized = len(data) != original_size

    encoded = base64.b64encode(data).decode("utf-8")
    name = os.path.splitext(os.path.basename(path))[0]
    return {
        "name": name,
        "image": {
            "type": "image",
            "value": f"data:{mime_type};base64,{encoded}",
        },
    }, was_resized


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"}


def collect_images(image_dir: str) -> list[str]:
    paths = []
    for entry in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(entry)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            paths.append(os.path.join(image_dir, entry))
    return paths


def open_output(path: str, fmt: str):
    if fmt == "bz2":
        return bz2.open(path, "wb")
    elif fmt == "gz":
        return gzip.open(path, "wb")
    else:
        return open(path, "wb")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_dir", help="Directory containing images")
    parser.add_argument("--output", "-o", help="Output file path (default: auto-named in cwd)")
    parser.add_argument(
        "--format",
        choices=["bz2", "gz", "jsonl"],
        default="bz2",
        help="Compression format (default: bz2)",
    )
    args = parser.parse_args()

    image_dir = os.path.abspath(args.image_dir)
    if not os.path.isdir(image_dir):
        print(f"ERROR: {image_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    images = collect_images(image_dir)
    if not images:
        print(f"ERROR: no images found in {image_dir}", file=sys.stderr)
        sys.exit(1)

    dir_name = os.path.basename(image_dir).lower().replace(" ", "-")
    ext_map = {"bz2": ".json.bz2", "gz": ".json.gz", "jsonl": ".jsonl"}
    output_path = args.output or os.path.join(os.getcwd(), f"{dir_name}-documents{ext_map[args.format]}")

    print(f"Processing {len(images)} images from {image_dir}")
    print(f"Output: {output_path}")

    uncompressed_bytes = 0
    doc_count = 0
    resized_count = 0

    with open_output(output_path, args.format) as out:
        for i, path in enumerate(images, 1):
            try:
                doc, was_resized = encode_image(path)
            except Exception as exc:
                print(f"  SKIP {os.path.basename(path)}: {exc}", file=sys.stderr)
                continue

            if was_resized:
                resized_count += 1

            line = (json.dumps(doc, separators=(",", ":")) + "\n").encode("utf-8")
            uncompressed_bytes += len(line)
            out.write(line)
            doc_count += 1

            if i % 50 == 0 or i == len(images):
                pct = i / len(images) * 100
                tag = " (resized)" if was_resized else ""
                print(f"  [{i}/{len(images)}] {pct:.0f}%  {os.path.basename(path)}{tag}", flush=True)

    compressed_bytes = os.path.getsize(output_path)

    print()
    print(f"Done. {resized_count}/{doc_count} images were resized to fit under 5 MB.")
    print()
    print("Add this to your track.json corpora section:")
    print("-" * 60)
    print(
        json.dumps(
            {
                "name": "semantic_image",
                "base-url": "https://rally-tracks.elastic.co/semantic_image",
                "documents": [
                    {
                        "source-file": os.path.basename(output_path),
                        "document-count": doc_count,
                        "compressed-bytes": compressed_bytes,
                        "uncompressed-bytes": uncompressed_bytes,
                    }
                ],
            },
            indent=2,
        )
    )
    print("-" * 60)


if __name__ == "__main__":
    main()
