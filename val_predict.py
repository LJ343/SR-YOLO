from ultralytics import YOLO

# 加载自己训练好的模型，填写相对于这个脚本的相对路径或者填写绝对路径均可
model = YOLO("runs/detect/best.pt")
data = "MyDatas/Dangerous Driving behavior dataset/data-driving.yaml"

# 开始进行验证，验证的数据集为'A_my_data.yaml'，图像大小为640，批次大小为4，置信度分数为0.25，交并比的阈值为0.6，设备为0，关闭多线程（windows下使用多线程加载数据容易出现问题）
#model.val(data=data, imgsz=640, batch=8, conf=0.001, iou=0.6, device="0", workers=0,project="runs/detect/BadODD_v8m/val",name="exp",save=True)
model.predict(source="E:/论文/YOLO系列/要用的数据/mobile detection.v1i.yolov11/valid/images/image2123_png.rf.5ddf15b4d6428dbd55112ddc5064d380.jpg",project="runs/detect/predict",name="drivi-v11",save=True)

