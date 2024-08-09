from ultralytics import YOLO

model = YOLO('model/best.pt')

result = model.predict('input_video/cricket.mp4', save=True)

print(result[0])

for box in result[0].boxes:
    print(box)