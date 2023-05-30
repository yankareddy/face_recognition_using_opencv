import cv2, numpy, os
haar_file  = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)
datasets = "dataset1"
print('Training...')
(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs,files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id +=1

(images, labels)= [numpy.array(lis) for lis in [images, labels]]
(width, height) = (130,100)
model = cv2.face.LBPHFaceRecognizer_create()
#model = cv2.face.FisherFaceRecognizer_create()

model.train(images, labels)
webcam = cv2.VideoCapture(1)
cnt=0

while True:
    _,img = cam.read()
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(grayImg,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
        face = gray[y:y+h,x:x+w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)
        cv2.reactangle(im, (x,y), (x+w, y+h), (0,255,0), 3)
        if prediction[1]<800:
            cv2.putText(im, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,2, (0,0,255))
            print(names[pridection[0]])
            cnt=0
        else:
            cnt+=1
            cv2.putText(im, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,255))
            if(cnt>100):
                print("Unknown Person")
                cv2.imwrite("unknown.jpg",im)
                cnt=0
    cv2.imshow('FaceRecognition', im)
    key = cv2.waitKey(10)
    if key == 'q':
        break
webcam.release()
cv2.destroyAllWindows()

            
