import cv2,os
alg = "haarcascade_frontalface_default.xml"
haar = cv2.CascadeClassifier(alg)
dataset='datasets'
subdata='varun'
path = os.path.join(dataset,subdata)
if not os.path.isdir(path):
    os.mkdir(path)
(w,h)=(130,100)
count = 1
vs =cv2.VideoCapture(0)
while count<31:
    _,img=vs.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = haar.detectMultiScale(gray,1.3,4)
    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+y,w+h),(0,255,0),2)
        face = gray[y:y+h,x:x+w]
        resized = cv2.resize(face,(w,h))
        cv2.imwrite("%s/%s.png" % (path,count),resized)
    count+=1
    cv2.imshow("dataset",img)
    key=cv2.waitKey(1)
    if(key==27):
        break
vs.release()
cv2.destroyAllWindows()

        
    
