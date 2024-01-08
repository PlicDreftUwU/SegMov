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
           ,19:'sofa',20:'train',21:'truck',22:'sign',23:'Rat'}
#Load model
net = cv.dnn.readNetFromCaffe(prototxt, model)
#--------- Read images ---------- #
cap = cv.VideoCapture('imgs\muestra.mp4')
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    height, width, _ = frame.shape
    frame_resized = cv.resize(frame, (300,300))
    #create blob
    blob = cv.dnn.blobFromImage(frame_resized, 0.007843, (300,300),(127.5,127.5,127.5))
    #print("blob.shape", blob.shape)
    #--------- Read images ---------- #
    net.setInput(blob)
    detections = net.forward()
    for detection in detections[0][0]:
        #print(detection)
        if detection[2] > 0.50:
            label = classes[detection[1]]
            box = detection[3:7] * [width, height, width, height]
            x_start , y_start, x_end, y_end = int(box[0]),int(box[1]),int(box[2]),int(box[3])
            cv.rectangle(frame,(x_start,y_start),(x_end, y_end), (0,255,0),2)
            cv.putText(frame, 'conf: {:.2f}'.format(detection[2] * 100), (x_start, y_start -5), 1, 1.2, (255,0,0),2 )
            cv.putText(frame, label, (x_start, y_start - 25), 1, 1.5, (0, 255,255),2)
    cv.imshow("Frame", frame)
    if cv.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv.destroyAllWindows()