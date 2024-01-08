import cv2 as cv
#--------- READ DNN MODEL ---------- #
# Cargamos arquitectura
prototxt = "model\MobileNetSSD_deploy.prototxt.txt"
#weights 
model = 'model\MobileNetSSD_deploy.caffemodel'
# Cargamos las clases
classes = {0:'background',1:'aeroplane',2:'bicycle',3:'bird'
           ,4:'boat',5:'bottle',6:'bus',7:'car',8:'cat',9:'chair'
           ,10:'cow',11:'dinigtable',12:'dog',13:'horse',14:'motorbike',
           15:'person',16:'phone',17:'plant',18:'sheep'
           ,19:'sofa',20:'train',21:'truck',23:'Rat'}
#Load model
net = cv.dnn.readNetFromCaffe(prototxt, model)
#--------- Read images ---------- #
image = cv.imread("imgs/rata.jpg")
height, width, _ = image.shape
image_resized = cv.resize(image,(300,300))
#Create blob
blob = cv.dnn.blobFromImage(image_resized, 0.00783, (300,300),(127.5,127.5,127.5))
print("blob.shape", blob.shape)
#--------- Read images ---------- #
net.setInput(blob)
detections = net.forward()
for detection in detections[0][0]:
    print(detection)
    if detection[2] > 0.50:
        label = classes[detection[1]]
        box = detection[3:7] * [width, height, width, height]
        x_start , y_start, x_end, y_end = int(box[0]),int(box[1]),int(box[2]),int(box[3])
        cv.rectangle(image,(x_start,y_start),(x_end, y_end), (0,255,0),2)
        cv.putText(image, 'conf: {:.2f}'.format(detection[2] * 100), (x_start, y_start -5), 1, 1.2, (255,0,0),2 )
        cv.putText(image, label, (x_start, y_start - 25), 1, 1.2, (255, 0,0),2)
cv.imshow('imagen', image)
cv.waitKey(0)
cv.destroyAllWindows()