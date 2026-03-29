import shutil
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO


IMAGE_DIR = "./images"
OUTPUT_ROOT = "./outputs"

MODEL_NAME = "yolo26n.pt"   # yolo8n.pt/yolo11n.pt/yolo26n.pt；yolo26n.pt表现最佳
CONF_THRESHOLD = 0.25       # threshold = 0.25为最佳，yolo26n下共计48个目标
USE_THRESHOLD_FOLDER = True

# COCO 数据集里 airplane 的类别 id 是 4
AIRPLANE_CLASS_ID = 4

# 支持的图片后缀
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_output_dir(output_root: str, model_name: str, conf_threshold: float, use_threshold_folder: bool) -> Path:
    model_stem = Path(model_name).stem

    if use_threshold_folder:
        threshold_folder = f"threshold_{conf_threshold:.2f}"
        output_dir = Path(output_root) / model_stem / threshold_folder
    else:
        output_dir = Path(output_root) / model_stem

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_image_files(image_dir: str):
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    image_files = sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    )
    return image_files

# Result picture
def draw_boxes_and_text(img: Image.Image, boxes_to_draw, count: int, model_name: str) -> Image.Image:
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)

    for box, score in boxes_to_draw:
        x1, y1, x2, y2 = box

        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

        text = f"{score:.2f}"
        text_y = max(0, y1 - 15)
        draw.text((x1, text_y), text, fill="green")

    info_text = f"Count: {count} \nModel: {model_name}"

    text_x = 10
    text_y = max(0, draw_img.height - 30)

    draw.text((text_x, text_y), info_text, fill="white")

    return draw_img


def main():
    # Load the model
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    print(f"[INFO] Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    image_files = get_image_files(IMAGE_DIR)
    if not image_files:
        print(f"[WARN] 在 {IMAGE_DIR} 中没有找到图片。")
        return

    output_dir = build_output_dir(
        output_root=OUTPUT_ROOT,
        model_name=MODEL_NAME,
        conf_threshold=CONF_THRESHOLD,
        use_threshold_folder=USE_THRESHOLD_FOLDER
    )
    print(f"[INFO] Output dir: {output_dir}")

    total_planes = 0

    for img_path in image_files:
        print(f"[INFO] Processing: {img_path.name}")

        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESHOLD,
            verbose=False,
            device=device
        )

        img = Image.open(img_path).convert("RGB")

        boxes_to_draw = []
        count = 0

        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()

            for box, score, cls_id in zip(xyxy, confs, clss):
                if int(cls_id) == AIRPLANE_CLASS_ID:
                    x1, y1, x2, y2 = map(int, box)
                    boxes_to_draw.append(((x1, y1, x2, y2), float(score)))
                    count += 1

        total_planes += count

        draw_img = draw_boxes_and_text(
            img=img,
            boxes_to_draw=boxes_to_draw,
            count=count,
            model_name=Path(MODEL_NAME).stem
        )

        save_path = output_dir / img_path.name
        draw_img.save(save_path)
        print(f"[INFO] Saved: {save_path} | Count: {count}")

    print("=" * 50)
    print(f"[DONE] Total images: {len(image_files)}")
    print(f"[DONE] Total airplanes detected: {total_planes}")
    print(f"[DONE] Results saved to: {output_dir}")


if __name__ == "__main__":
    main()