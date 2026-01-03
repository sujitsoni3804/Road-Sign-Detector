from ultralytics import YOLO

def main():
    model = YOLO('yolov8n-cls.yaml')
    results = model.train(
        data='gtsrb.yaml',
        imgsz=288,
        epochs=50,
        batch=16,
        lr0=0.01,
        optimizer='SGD',
        momentum=0.9,
        weight_decay=5e-4,
        patience=10,
        verbose=True,
        save=True,
        project='runs/classification',
        name='yolov8_custom_cls_experiment'
    )
    print("Training complete!")
    model.val()

if __name__ == "__main__":
    main()
