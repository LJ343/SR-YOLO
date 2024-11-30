from ultralytics import YOLO

# Load a model
#model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("E:/PyCharm/python3.9.13/Project/YOLOv5sPlus/yolov5-master/runs/train/exp35/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx",project="runs/detect/export",name="v5s-onnx")