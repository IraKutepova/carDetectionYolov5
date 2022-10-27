import torch
import cv2 as cv
import time
# using YoloV5 model for car detection
# with internet connection
# model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
# local
model = torch.hub.load('yolov5', 'yolov5n', source='local')
times = []
x_resize, y_resize = 1280, 640

# Car detection on video
cap = cv.VideoCapture('testvideo.mp4') 
while True:
    success, frame = cap.read()
    if not success:
        print("Can't read the file!")
        break
    frame = cv.resize(frame,(x_resize, y_resize))
    start = time.time()
    result = model(frame)
    df = result.pandas().xyxy[0]
    # loop for all objects finded on video not only cars. it can be person or bus or boats
    for ind in df.index: 
        x1, y1 = int(df['xmin'][ind]), int(df['ymin'][ind])
        x2, y2 = int(df['xmax'][ind]), int(df['ymax'][ind])
        if df['name'][ind] == 'car' and int(df['confidence'][ind] * 100) > 50 : # only cars detection
 
             label = f"{df['name'][ind]} {int(df['confidence'][ind] * 100)}%"
             cv.rectangle(frame, (x1,y1), (x2,y2), (255,255,0), 2)
             cv.putText(frame, label, (x1,y1-5), cv.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
    end = time.time()
    # print('time to detect:', round(end-start,3))

    times.append(round(end-start,3))

    cv.imshow('Video', frame)
    k=cv.waitKey(1)
    if k==ord('q'):
        break
if times:
    print('Average time to detect:', sum(times) / len(times))
    print(f'min time to detect: {min(times)}; max time to detect: {max(times)}')

# очистка
cap.release()
cv.destroyAllWindows()
