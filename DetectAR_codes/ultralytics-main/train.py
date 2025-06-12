from ultralytics import YOLO

# Load a model
model = YOLO("DetectAR/DetectAR_codes/models/yolov8-ecasdcn.yaml").load("./yolov8s.pt")  # build a new model from scratch
model.train(data="/DetectAR/DetectAR_data/AR.yaml", epochs=100, batch=16, imgsz=1024, patience=50, single_cls=True, optimizer='AdamW', lr0=0.001, resume=True)  # train the model
metrics = model.val()
