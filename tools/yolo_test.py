from pathlib import Path
import argparse

from PIL import Image
import torch
from ultralytics import YOLO

try:
    import rawpy
except ImportError:
    rawpy = None


def resize_and_pad(
    img: Image.Image, imgsz: int, color=(114, 114, 114)
) -> tuple[Image.Image, float, int, int]:
    w, h = img.size
    scale = imgsz / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    padded = Image.new("RGB", (imgsz, imgsz), color)
    offset = ((imgsz - new_w) // 2, (imgsz - new_h) // 2)
    padded.paste(img, offset)
    return padded, scale, offset[0], offset[1]


def prepare_image(
    input_path: str, output_dir: str = "prepared", imgsz: int = 1280
) -> tuple[Path, Image.Image, float, int, int]:
    src = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}_{imgsz}.jpg"

    if src.suffix.lower() == ".nef":
        if rawpy is None:
            raise ImportError("rawpy is required for .nef files. Install with: pip install rawpy")
        with rawpy.imread(str(src)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
            )
        img = Image.fromarray(rgb)
    else:
        img = Image.open(src).convert("RGB")

    resized, scale, pad_x, pad_y = resize_and_pad(img, imgsz)
    resized.save(out_path, "JPEG", quality=95, optimize=True)
    return out_path, img, scale, pad_x, pad_y


def iter_input_files(source: str, recursive: bool = False) -> list[Path]:
    src = Path(source)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".nef"}
    if src.is_dir():
        pattern = "**/*" if recursive else "*"
        files = [p for p in src.glob(pattern) if p.suffix.lower() in exts and p.is_file()]
        return sorted(files)
    return [src]


def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def save_crops(
    img: Image.Image,
    boxes_xyxy: list[list[float]],
    confs: list[float],
    classes: list[int],
    names: dict[int, str],
    scale: float,
    pad_x: int,
    pad_y: int,
    out_dir: Path,
    base_name: str,
    max_crops: int = 3,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = img.size
    saved = 0
    items = list(zip(boxes_xyxy, confs, classes))
    items.sort(key=lambda x: x[1], reverse=True)
    for i, (box, conf, cls_id) in enumerate(items[:max_crops]):
        x1, y1, x2, y2 = box
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        x1 = clamp(x1, 0, w)
        y1 = clamp(y1, 0, h)
        x2 = clamp(x2, 0, w)
        y2 = clamp(y2, 0, h)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
        label = names.get(cls_id, f"class_{cls_id}")
        safe_label = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
        crop_path = out_dir / f"{base_name}_{safe_label}_{conf:.3f}_{i:03d}.jpg"
        crop.save(crop_path, "JPEG", quality=95, optimize=True)
        saved += 1
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO inference with optional RAW preprocessing.")
    parser.add_argument("source", help="Path to input image or a folder of images.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference size (square).")
    parser.add_argument("--device", default="mps", help="Device to use: mps/cpu.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When source is a folder, scan subfolders too.",
    )
    parser.add_argument(
        "--prepared-dir",
        default="prepared",
        help="Directory for converted/resized images.",
    )
    parser.add_argument(
        "--crops-dir",
        default="crops",
        help="Directory where cropped detections are saved.",
    )
    args = parser.parse_args()

    print("MPS available:", torch.backends.mps.is_available())

    model = YOLO("yolo11n.pt")
    inputs = iter_input_files(args.source, recursive=args.recursive)
    if not inputs:
        raise SystemExit("No supported images found in the provided path.")

    for input_path in inputs:
        prepared, original_img, scale, pad_x, pad_y = prepare_image(
            str(input_path),
            output_dir=args.prepared_dir,
            imgsz=args.imgsz,
        )
        results = model.predict(
            source=str(prepared),
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
        )
        # results is a list of Results objects
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None:
            continue
        boxes_xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []
        classes = boxes.cls.cpu().tolist() if boxes.cls is not None else []
        if not boxes_xyxy or not confs or not classes:
            continue
        crop_dir = Path(args.crops_dir) / input_path.stem
        saved = save_crops(
            original_img,
            boxes_xyxy,
            confs,
            [int(c) for c in classes],
            results[0].names,
            scale,
            pad_x,
            pad_y,
            crop_dir,
            input_path.stem,
            max_crops=3,
        )
        print(f"{input_path.name}: saved {saved} crops to {crop_dir}")


if __name__ == "__main__":
    main()
