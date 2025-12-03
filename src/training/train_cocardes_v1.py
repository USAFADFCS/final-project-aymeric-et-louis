from ultralytics import YOLO


def main():
    # Charger un modèle pré-entraîné (yolov8n)
    model = YOLO("../../models/yolo/yolov8n.pt")

    # Entraînement
    model.train(
        data=r"C:\Users\demeu\PycharmProjects\final-project-aymeric-et-louis\CocardesV2.v2i.yolov8\data.yaml",
        epochs=60,
        imgsz=640,
        batch=16,
        device=0,  # GPU 0
        name="cocardes_v1",
    )


if __name__ == "__main__":
    main()
