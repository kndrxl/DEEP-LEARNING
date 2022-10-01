
import cv2
import numpy as np
import imutils
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
model = load_model("cnn_iteration_2.model")
image_path = "images/"

def image_inference(model, img_path):
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.array(image, dtype="float32")
    y_pred = model.predict(image.reshape(1, 224, 224, 3))
    print(y_pred)
    y_pred = y_pred > 0.5
    if(y_pred == 0):
        pred = 'man'
    else:
        pred = 'woman'
    return pred

def detect_face_and_gender(frame, faceNet, model):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = model.predict(faces, batch_size=32)
    return (locs, preds)

def inference():
    vid = cv2.VideoCapture(0)
    while True:
        _, frame = vid.read()
        frame = imutils.resize(frame, width=600)
        (locs, preds) = detect_face_and_gender(frame, faceNet, model)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            print(pred)
            label = "Men" if pred[0] <0.5 else "Women"
            color = (255, 255, 255) if label == "Women" else (0, 0, 255)
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(2) & 0xFF
        if key == 27:
            break
    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Starting Inference Phase...")
    inference()