import os.path
import cv2
import boto3
import re
from PIL import Image
from cvu.detector.yolov5 import Yolov5
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
import tensorflow as tf
from pyzbar.pyzbar import decode
import var
import time
import datetime
from datetime import date
import logging
from logging.handlers import TimedRotatingFileHandler
from sanofi import predictSanofi 
from demo import predictDemo 

app = FastAPI()
origins = [
    "http://localhost", "http:localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IC_CLASS_NAMES = ["sanofi_001", "sanofi_002", "sanofi_003", "sanofi_004", "sanofi_005","sanofi_007"]

# TUBE_OD_CLASS_NAMES = ["Label_IN", "Tube", "Labels", "Handwritten", "base"]  # Ml model for crop and predict
TUBE_OD_CLASS_NAMES = ["Label", "Tube", "base"]  # Ml model for crop and predict

TUBE_MODEL_OD = Yolov5(classes=TUBE_OD_CLASS_NAMES, backend="onnx", weight="models/sanofiModel/tube/110424.onnx", device="cpu")

#MODEL_IC = tf.keras.models.load_model("models/IC/model0603.h5")
MODEL_IC_crop = tf.keras.models.load_model("models/sanofiModel/IC/sanofi1_0.h5")
AWS_TEXTRACT = boto3.client('textract', aws_access_key_id=var.aws_access_key_id,
                            aws_secret_access_key=var.aws_secret_access_key,
                            region_name=var.region)

logpath = datetime.datetime.now().strftime("%Y-%m-%d") + ".log"
print(logpath)
log_file = os.path.join("enrichlogs", logpath)
handler = TimedRotatingFileHandler(filename=log_file,
                                   when="midnight",
                                   interval=1,
                                   backupCount=31)
formater = logging.Formatter("%(asctime)s %(thread)s | %(levelname)s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formater)
handler.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(handler)
logging.info("log initiated")


@app.get("/ping")  # to test ping on api
async def ping():
    logging.info("API is active")
    return "Hello, I am alive"


@app.get("/logdaterange")
async def logdaterange(
        start: date = Query(default=None, example="2024-02-22"), end: date = Query(default=None, example="2024-02-22")):
    if date is None:
        return "Date parameter is required", 400
    log_lines = []
    current_date = start
    while current_date <= end:
        log_file_path = f'enrichlogs/{(current_date)}.log'
        try:
            with open(log_file_path, 'r') as file:
                log_lines.extend(file.readlines())
        except FileNotFoundError:
            print(f"File {log_file_path} not found.")
            logging.warning("No log for this day")
        current_date += datetime.timedelta(days=1)
    return log_lines


@app.post("/predict")
# this method takes care of classification, object detection, roi croping, rotation if required, and AWS text extraction
async def predict(
        model:str,
        file: UploadFile = File(...)):
    try:
        if model == "sanofi":
            return await predictSanofi(logging, file);
        return await predictDemo(logging, file);
    except Exception as e:
        logging.exception(e)
        return {
            'Label': "New Image: Input image is not been trained or wrong image",
            'Confidence': 0.0,
            'Barcode': "",
            'Handwriten': [],
            'Text': [],
            'FormatedText': {}
         }


@app.post("/crop")
async def crop(
        file: UploadFile = File(...)):
    imageapi = await file.read()
    imageapiOD = Image.open(BytesIO(imageapi))
    # for classification
    image_array = np.frombuffer(imageapi, np.uint8)
    image_decode = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image_resized = cv2.resize(image_decode, (512, 512))
    image_dims = np.expand_dims(image_resized, axis=0)
    predictions = MODEL_IC_crop.predict(image_dims)
    predicted_class = IC_CLASS_NAMES[np.argmax(predictions[0])]
    logging.info(predicted_class)

    filename, file_extension = os.path.splitext(file.filename)
    print("filename:", filename + file_extension)
    logging.info(filename + file_extension)
    if file_extension == ".png":
        img_format = 'PNG'
        media_type_name = "image/png"
    elif file_extension == ".jpeg":
        img_format = 'JPEG'
        media_type_name = "image/jpg"
    elif file_extension == ".jpg":
        img_format = 'JPEG'
        media_type_name = "image/jpg"
    image = np.array(imageapiOD.convert('RGB'))
    predictions_OD = TUBE_MODEL_OD(image)  # results of OD
    print("Detection", predictions_OD)

    det_list = []
    # loop through
    for prediction in predictions_OD:
        det_dict = {}
        x1 = int(prediction.bbox[0])
        y1 = int(prediction.bbox[1])
        x2 = int(prediction.bbox[2])
        y2 = int(prediction.bbox[3])
        croped = image[y1:y2, x1:x2]
        # cv2.imwrite("output.jpg", croped)
        height, width, _ = croped.shape
        # print(height, width)
        size = width * height
        det_dict["class"] = prediction.class_name
        det_dict["im"] = croped
        det_dict["size"] = size
        det_list.append(det_dict)
    try:
        max_dict = max(det_list, key=lambda x: x["size"])
        max_size = max_dict["size"]
        max_image = max_dict["im"]
        best_image = np.array(max_image)
        image1 = Image.fromarray(best_image)
        max_class = max_dict["class"]
        print("actual class", max_class)
        logging.info(max_class)

        if max_class == "Label_IN":
            image1 = image1.rotate(180)  # rotating
            print("image is been rotated upside-down")

        if predicted_class == "Label3":
            image1 = image1.rotate(-90, expand=True)  # rotating
            print("image is been rotated clockwise")

        if predicted_class == "Label5":
            image1 = image1.rotate(90, expand=True)  # rotating
            print("image is been rotated anticlockwise")

        imgByteArr = BytesIO()
        image1.save(imgByteArr, format=img_format)
        imgByteArr.seek(0)
        logging.info("cropping is Succesfully")
        return StreamingResponse(content=imgByteArr, media_type=media_type_name)
    except:
        logging.error(HTTPException)
        raise HTTPException(status_code=404, detail="Item not found")


@app.post("/bounding")
async def bounding(
        file: UploadFile = File(...)):
    imageapi = Image.open(BytesIO(await file.read()))
    filename, file_extension = os.path.splitext(file.filename)
    print("file_extension : ", file_extension)
    logging.info(file_extension)
    if file_extension == ".png":
        img_format = 'PNG'
        media_type_name = "image/png"
    elif file_extension == ".jpeg":
        img_format = 'JPEG'
        media_type_name = "image/jpg"
    elif file_extension == ".jpg":
        img_format = 'JPEG'
        media_type_name = "image/jpg"
    image = np.array(imageapi.convert('RGB'))
    predictions_OD = TUBE_MODEL_OD(image)  # results of OD
    # predictions = MODEL_OD(image)
    print(predictions_OD)
    logging.info(predictions_OD)
    predictions_OD.draw(image)
    best_image = np.array(image)
    image1 = Image.fromarray(best_image)
    imgByteArr = BytesIO()
    image1.save(imgByteArr, format=img_format)
    imgByteArr.seek(0)
    logging.info("bounding coordinates received")
    return StreamingResponse(content=imgByteArr, media_type=media_type_name)
    # return StreamingResponse(content=imgByteArr, media_type=media_type_name)


if __name__ == "__main__":
    #uvicorn.run(app, host='127.0.0.1', port=5000)
    uvicorn.run(app, host='0.0.0.0', port=8000)
