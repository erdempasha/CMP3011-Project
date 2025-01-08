import ultralytics as ul

if __name__ == '__main__':

    model = ul.YOLO("yolo11l.pt") # pretrained model for further training

    results = model.train(
        data    = "./processed/dataset.yaml",
        epochs  = 100,
        imgsz   = 640,
        batch   = -1,
        augment = True
    )