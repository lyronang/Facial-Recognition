import cv2
import dlib
from cntk.ops.functions import load_model
from keras.models import load_model
import numpy as np
import keras

model = load_model("C:\\Users\\lyron\\Desktop\\faceid\\faceid\\Best-weights-my_model-002-0.0034-0.9993.h5")
model.summary()

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
constant = 5
Names_list = ['Amirah', 'CJ', 'Ernest', 'Jacky', 'Jason', 'Jiayi', 'Joab', 'Kaiwen', 'LiZhe', 'Shank']

while(True):
    ret, frame = cap.read()

    try:
        dets = detector(frame, 1)
        
        left = max(dets[0].left() - constant, 0)
        right = min(dets[0].right()+constant, 640)
        top = max(0, dets[0].top()-constant)
        bottom = min(480, dets[0].bottom()+constant)
        face_image = frame[top:bottom, left:right, :]
        cv2.rectangle(frame, (left, top), (right,bottom), (0,255,0),3)
        input_img=cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB).astype("float32")
        input_img_resize=cv2.resize(input_img,(64,64))
        #input_img_resize /= 255.
        #input_img_resize=np.expand_dims(input_img_resize, axis=-1)
        #input_img_resize=np.expand_dims(input_img_resize, axis=0)
        norm_input_img = keras.applications.vgg19.preprocess_input(input_img_resize)
        norm_input_img=np.expand_dims(norm_input_img, axis=0) 
        prediction = model.predict(norm_input_img)
        #prediction = np.squeeze(model.eval({model.arguments[0]:[input_img_resize]}))
        top_class = np.argmax(prediction[0])
        text = Names_list[top_class]
        cv2.putText(frame, text , (180,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        print(top_class)
        cv2.imshow("face recognition", frame)
    except:
        print("fail")
    


    """
    input_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32")
    input_img_resize=cv2.resize(input_img,(128,128))
    input_img_resize=np.expand_dims(input_img_resize, axis=-1)
    input_img_resize /= 255.
    prediction = np.squeeze(model.eval({model.arguments[0]:[input_img_resize]}))
    top_class = np.argmax(prediction)
    print(top_class)

    cv2.imshow("face recognition", frame)
    """
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
