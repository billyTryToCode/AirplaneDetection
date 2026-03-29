import os
import cv2
import torch
from PIL import Image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)

def main():
    input_dir = "images"
    output_dir = "outputs"
    score_thresh = 0.4  # 可改成 0.3 / 0.5 试效果

    os.makedirs(output_dir, exist_ok=True)

    # 1. 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 加载模型与权重
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    model.to(device)
    model.eval()

    # 3. 官方预处理
    preprocess = weights.transforms()

    # 4. 类别表
    categories = weights.meta["categories"]

    # 5. 支持的图片格式
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    image_names = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    )

    if not image_names:
        print(f"No images found in: {input_dir}")
        return

    print(f"Found {len(image_names)} image(s).")

    for name in image_names:
        img_path = os.path.join(input_dir, name)
        print(f"Processing: {img_path}")

        # 用 OpenCV 读图，后面便于画框和保存
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"Skip unreadable image: {img_path}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # 预处理后送入模型
        img_tensor = preprocess(pil_img).to(device)

        with torch.no_grad():
            prediction = model([img_tensor])[0]

        boxes = prediction["boxes"].detach().cpu()
        labels = prediction["labels"].detach().cpu()
        scores = prediction["scores"].detach().cpu()

        airplane_count = 0

        for box, label, score in zip(boxes, labels, scores):
            cls_name = categories[label.item()]
            conf = score.item()

            # 只保留 airplane
            if cls_name != "airplane":
                continue

            if conf < score_thresh:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())

            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                bgr,
                f"{cls_name} {conf:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            airplane_count += 1

        out_path = os.path.join(output_dir, name)
        cv2.imwrite(out_path, bgr)
        print(f"Saved: {out_path} | airplanes detected: {airplane_count}")

    print("Done.")

if __name__ == "__main__":
    main()