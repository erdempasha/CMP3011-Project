import ultralytics as ul

if __name__ == '__main__':

    model = ul.YOLO("./models/best.pt")

    model.predict(
        source=0,
        show=True,
        conf=0.55
    ) #webcam prediction

    model.predict(
        source='https://youtu.be/etZK-GrUYgM',
        show=True,
        conf=0.55
    ) #example video prediction