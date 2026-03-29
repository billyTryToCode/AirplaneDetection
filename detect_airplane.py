import os
import shutil
import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)


def main():
    input_dir = "images"
    output_root = "outputs"
    score_thresh = 0.3   # threshold = 0.3为最佳，共计46个目标

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "fasterrcnn_resnet50_fpn_v2"

    output_dir = os.path.join(output_root, f"thresh_{score_thresh:.2f}")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Loading model
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device)
    model.eval()

    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    total_airplane_counts = 0

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)

        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(name)[1].lower()
        if ext not in valid_exts:
            continue

        img = Image.open(path).convert("RGB")
        img_tensor = preprocess(img).to(device)

        with torch.no_grad():
            pred = model([img_tensor])[0]

        count = 0
        boxes_to_draw = []

        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            if score.item() < score_thresh:
                continue

            if categories[label.item()] == "airplane":
                count += 1
                total_airplane_counts += 1
                boxes_to_draw.append((box.cpu().tolist(), score.item()))

        print(f"{name} -> airplanes: {count}")

        # Result picture
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)

        for box, score in boxes_to_draw:
            x1, y1, x2, y2 = box

            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

            text = f"{score:.2f}"
            draw.text((x1, max(0, y1 - 15)), text, fill="green")

        info_text = f"Count: {count} \nModel: {model_name}"

        text_x = 10
        text_y = draw_img.height - 30

        draw.text((text_x, text_y), info_text, fill="white")

        # Save the result
        save_path = os.path.join(output_dir, name)
        draw_img.save(save_path)

    print(f"Airplanes in total: {total_airplane_counts}")
    print(f"Annotated images saved to: {output_dir}")


if __name__ == "__main__":
    main()