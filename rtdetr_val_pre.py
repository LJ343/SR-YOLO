from ultralytics import RTDETR

# 加载自己训练好的模型，填写相对于这个脚本的相对路径或者填写绝对路径均可
model = RTDETR("runs/detect/BadODD_rtdetrx/train/exp1/weights/last.pt")
data = "./MyDatas/dlenigma1/BadODD_data.yaml"

# 开始进行验证，验证的数据集为'A_my_data.yaml'，图像大小为640，批次大小为4，置信度分数为0.25，交并比的阈值为0.6，设备为0，关闭多线程（windows下使用多线程加载数据容易出现问题）
model.val(data=data, imgsz=640, batch=8, conf=0.001, iou=0.6, device="0", workers=0,project="runs/detect/BadODD_rtdetrx/val",name="exp",save=True)
#model.predict(source="MyDatas/dlenigma1/BadODD/images/test/chittagong_bohoddarhat1_3894.jpg",project="runs/detect/predict",name="1",save=True)

