import os
import torch
from PIL import Image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)

def main():
    input_dir = "images"
    score_thresh = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device)
    model.eval()

    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    totalAirplaneCounts = 0

    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)

        img = Image.open(path).convert("RGB")
        img_tensor = preprocess(img).to(device)

        with torch.no_grad():
            pred = model([img_tensor])[0]

        count = 0

        for label, score in zip(pred["labels"], pred["scores"]):
            if score < score_thresh:
                continue

            # 👉 关键：只统计 airplane
            if categories[label.item()] == "airplane":
                count += 1
                totalAirplaneCounts += 1

        print(f"{name} -> airplanes: {count}")

    print(f"Airplanes in total: {totalAirplaneCounts}")

if __name__ == "__main__":
    main()