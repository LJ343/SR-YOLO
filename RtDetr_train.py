#import warnings
#warnings.filterwarnings('ignore')

from ultralytics import RTDETR

if __name__ == '__main__':
	#model = RTDETR('rtdetr-x.pt')
	model = RTDETR("./MyDatas/dlenigma1/BadODD_rtdetr-x.yaml")
	data = "./MyDatas/dlenigma1/BadODD_data.yaml"
	# model.load('rtdetr-x.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度
	model.train(data=data,
	            # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
	            cache=True,
	            imgsz=640,
	            epochs=100,
	            single_cls=False,  # 是否是单类别检测
	            batch=-1,
	            pretrained=False,
	            patience=50,
	            save_period=25,
	            cos_lr=True,
	            close_mosaic=0,
	            val=True,   #培训期间不 验证/测试
	            resume=True,
	            lr0=0.001,
	            workers=0,
	            device='0',
	            optimizer='Adam',  # using SGD 优化器 默认为auto建议大家使用固定的.
	            # resume=, # 续训的话这里填写True, yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
	            amp=True,  # 如果出现训练损失为Nan可以关闭amp
	            project='runs/detect/BadODD_rtdetrx/train',
	            name='exp',
	            )
