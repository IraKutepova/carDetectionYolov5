import torch
import cv2 as cv
import time
# using YoloV5 model for car detection
# with internet connection
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# local
model = torch.hub.load('yolov5', 'yolov5n', source='local')

# Car detection on video
cap = cv.VideoCapture('video1.mp4') 
while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv.resize(frame,(1200,600))
    start = time.time()
    result = model(frame)
    df = result.pandas().xyxy[0]
    # loop for all objects finded on video not only cars. it can be person or bus or boats
    # for ind in df.index:
    #     x1, y1 = int(df['xmin'][ind]), int(df['ymin'][ind])
    #     x2, y2 = int(df['xmax'][ind]), int(df['ymax'][ind])
    #     label = f"{df['name'][ind]} {int(df['confidence'][ind] * 100)}%"
        
    #     cv.rectangle(frame, (x1,y1), (x2,y2), (255,255,0), 2)
    #     cv.putText(frame, label, (x1,y1-5), cv.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
    # only cars detection
    if 2 in df.index:
        ind = 2
        x1, y1 = int(df['xmin'][ind]), int(df['ymin'][ind])
        x2, y2 = int(df['xmax'][ind]), int(df['ymax'][ind])
        label = f"[{df['name'][ind]} {int(df['confidence'][ind] * 100)}%]"
        
        cv.rectangle(frame, (x1,y1), (x2,y2), (255,255,0), 2)
        cv.putText(frame, label, (x1,y1-5), cv.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
    end = time.time()
    print('time to detect:', round(end-start,3))
    
    cv.imshow('Video', frame)
    k=cv.waitKey(1)
    if k==ord('q'):
        break

# очистка
cap.release()
cv.destroyAllWindows()
