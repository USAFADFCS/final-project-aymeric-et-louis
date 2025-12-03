from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    data_yaml = ROOT / "data" / "processed" / "cocardes_merged_v2" / "data.yaml"

    model = YOLO("yolov8n.pt")  # base
    model.train(
        data=str(data_yaml),
        epochs=80,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        name="cocardes_merged_v2",
    )


if __name__ == "__main__":
    main()
