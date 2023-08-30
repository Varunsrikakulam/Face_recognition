import cv2,os,numpy
alg = "haarcascade_frontalface_default.xml"
haar =cv2.CascadeClassifier(alg)
print("training")
(width,height)=(130,100)
dataset='datasets'
(images,labels,name,id)=([],[],{},0)
for (subdir,dire,files) in os.walk(dataset):
    for subdir in dire:
        name[id]=subdir
        subpath = os.path.join(dataset,subdir)
        for filename in os.listdir(subpath):
            path = subpath+ '/' +filename
            label = id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id+=1
(images,labels)=[numpy.array(ele) for ele in [images,labels]]
model=cv2.face.LBPHFaceRecognizer_create()
model.train(images,labels)
vs = cv2.VideoCapture(0)
cnt=0
while True:
    (_,img)=vs.read()
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray,1.3,4)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+w),(0,255,255),2)
        face= gray[y:y+h,x:x+h]
        resize=cv2.resize(face,(width,height))
        prediction = model.predict(resize)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        if(prediction[1]<800):
            cv2.putText(img,"%s-%.0f" %(name[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
        else:
            cnt+=1
            cv2.putText(img,"unknown",(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
            if(cnt>100):
                cv2.imwrite("Unknown.png",img)
                cnt=0
    cv2.imshow("Facedetect",img)
    key =cv2.waitKey(10)
    if(key==27):
        break
vs.release()
cv2.destroyAllWindows()