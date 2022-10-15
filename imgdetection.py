import torch
import cv2 as cv
# using YoloV5 model for car detection
# with internet connection
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# local
model = torch.hub.load('yolov5', 'yolov5n', source='local')


# car detection on images
img = cv.imread('testcar.jpg')

result = model(img)

print(result)

df = result.pandas().xyxy[0]
print(df)

for ind in df.index:
    x1, y1 = int(df['xmin'][ind]), int(df['ymin'][ind])
    x2, y2 = int(df['xmax'][ind]), int(df['ymax'][ind])
    label = df['name'][ind]
    cv.rectangle(img, (x1,y1), (x2,y2), (255,255,0), 2)
    cv.putText(img, label, (x1,y1-5), cv.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)

cv.imshow('Image', img)
cv.waitKey(0)