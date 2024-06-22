import cv2
import boto3
import re
from PIL import Image
from cvu.detector.yolov5 import Yolov5
import numpy as np
from fastapi import  File, UploadFile
from io import BytesIO
import tensorflow as tf
from pyzbar.pyzbar import decode
import var
from datetime import date

IC_CLASS_NAMES = ["Label1", "Label2", "Label3", "Label4","Label5"]

TUBE_OD_CLASS_NAMES = ["Label_IN", "Tube", "Labels", "Handwritten", "base"]  # Ml model for crop and predict

TUBE_MODEL_OD = Yolov5(classes=TUBE_OD_CLASS_NAMES, backend="onnx", weight="models/demoModel/tube/3001.onnx", device="cpu")

MODEL_IC = tf.keras.models.load_model("models/demoModel/IC/model0603.h5")
AWS_TEXTRACT = boto3.client('textract', aws_access_key_id=var.aws_access_key_id,
                            aws_secret_access_key=var.aws_secret_access_key,
                            region_name=var.region)



async def predictDemo(
        logging, 
        file: UploadFile = File(...)):
    imageapi = await file.read()
    imageapiOD = Image.open(BytesIO(imageapi))
    image = np.array(imageapiOD.convert('RGB'))
    predictions_OD = TUBE_MODEL_OD(image)  # results of OD
    # for classification
    image_array = np.frombuffer(imageapi, np.uint8)
    image_decode = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image_resized = cv2.resize(image_decode, (512, 512))
    image_dims = np.expand_dims(image_resized, axis=0)
    print("imageshape:", image_dims.shape)
    predictions = MODEL_IC.predict(image_dims)
    print(predictions)
    score = (predictions[0])
    score = (100 * np.max(score))
    round_score = round(score, 2)
    predicted_class = IC_CLASS_NAMES[np.argmax(predictions[0])]
    print(predicted_class, score)
    print(predictions_OD)
    # predictions.draw(image)
    det_list = []
    # loop through
    for prediction in predictions_OD:
        # print info
        # det_dict={"class":[],"im":[],"size":[]}
        det_dict = {}
        x1 = int(prediction.bbox[0])
        y1 = int(prediction.bbox[1])
        x2 = int(prediction.bbox[2])
        y2 = int(prediction.bbox[3])
        croped = image[y1:y2, x1:x2]
        height, width, _ = croped.shape
        # print(height, width)
        size = width * height
        det_dict["class"] = prediction.class_name
        det_dict["im"] = croped
        det_dict["size"] = size
        det_list.append(det_dict)
        # print(det_list)
    found_handwritten = any(item.get("class") == "Handwritten" for item in det_list)
    if found_handwritten:
        try:
            # For handwritten
            max_imageHW = max(filter(lambda x: x["class"] == "Handwritten", det_list), key=lambda x: x["size"])
            best_imageHW = np.array(max_imageHW["im"])
            imagehw = Image.fromarray(best_imageHW)
            for odclass in det_list:
                if odclass["class"] == "Label_IN":
                    imagehw = imagehw.rotate(180)
                    # imagehw.show()
                    break
            print("handwritten image is been inverted ")
            print("*************************************************************")
            # imagehw.show()
            imgByteArr1 = BytesIO()
            imagehw.save(imgByteArr1, format='JPEG')
            imgByteArr1.seek(0)
            response = AWS_TEXTRACT.detect_document_text(Document={'Bytes': imgByteArr1.getvalue()})
            extracted_hwtext = []
            for item in response['Blocks']:
                if item['BlockType'] == 'LINE':
                    # extracted_hwtext = item
                    extracted_hwtext += item['Text'],
            # print(extracted_hwtext)
        except Exception as e:
            logging.exception(e)
            extracted_hwtext = []
    else:
        extracted_hwtext = []
    try:
        # for label image
        max_dict = max(det_list, key=lambda x: x["size"])
        max_size = max_dict["size"]
        max_image = max_dict["im"]
        # print("max_size" ,max_size)
        best_image = np.array(max_image)
        image1 = Image.fromarray(best_image)
        max_class = max_dict["class"]
        print("actual class :", max_class)
        if max_class == "Labels_IN":
            image1 = image1.rotate(180)  # rotating
            print("image is been rotated upside-down")
        if predicted_class =="Label3":
            image1 = image1.rotate(-90,expand = True)  # rotating
            print("image is been rotated clockwise")
        if predicted_class =="Label5":
            image1 = image1.rotate(90,expand = True)  # rotating
            print("image is been rotated anticlockwise")

        detectedBarcodes = decode(image1)
        if not detectedBarcodes:
            warning = "Barcode Not Detected or your barcode is blank/corrupted!"
            print(warning)
            Barcode = ""
        else:
            # Traverse through all the detected barcodes in image
            for barcode in detectedBarcodes:
                if barcode.data != "":
                    # Print the barcode data
                    print(barcode.data.decode('utf-8'))
                    Barcode = barcode.data.decode('utf-8')
        imgByteArr = BytesIO()
        image1.save(imgByteArr, format='JPEG')
        imgByteArr.seek(0)
        response = AWS_TEXTRACT.detect_document_text(Document={'Bytes': imgByteArr.getvalue()})
        # Extract text from the response
        extracted_text = []
        extracted_conf = []
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                # extracted_text += item['Text'] + '\n'
                extracted_text += item['Text'],
                extracted_conf += round(item['Confidence'], 2),
        # print(extracted_text)
        print(f"TEXT: {[extracted_text]},confidence : {[extracted_conf]}")
        image1.close()
        if predicted_class == "Label4":
            key = ["OriginatingID", "Study", "SUBLOT", "PATIENT"]
            formated_text = {}
            try:
                OriginatingIDformat = extracted_text[0].split(",", 1)
                OriginatingID = OriginatingIDformat[0]
                OriginatingIDconf = extracted_conf[0]
            except:
                OriginatingID = "Error"
                OriginatingIDconf = 0.0
            try:
                Studyformat = extracted_text[0].split(",", 1)
                Study = Studyformat[1]
                Studyconf = extracted_conf[0]
            except:
                Study = "Error"
                Studyconf = 0.0
            try:
                Sublotformat = extracted_text[1].split(" ", 1)
                Sublot = Sublotformat[1]
                Sublotconf = extracted_conf[1]
            except:
                Sublot = "Error"
                Sublotconf = 0.0
            try:
                Patientformat = extracted_text[2].split(" ", 1)
                Patient = Patientformat[1]
                Patientconf = extracted_conf[2]
            except:
                Patient = "Error"
                Patientconf = 0.0
                Sublot = extracted_text[1].split(" ", 1)
            exp_list = [OriginatingID, Study, Sublot, Patient]
            exp_conf = [OriginatingIDconf, Studyconf, Sublotconf, Patientconf]
            for i in range(0, 4):
                formated_text[key[i]] = {"Value": exp_list[i], "Conf": exp_conf[i]}
            print("label4", formated_text)
            logging.debug(formated_text)
            logging.info("extraction completed")
        elif predicted_class == "Label1":
            key = ["OriginatingID", "SubjectID", "Study", "Date"]
            formated_text = {}
            try:
                OriginatingIDFormat = re.findall(r':\s?(\S+)', extracted_text[0])
                OriginatingID = OriginatingIDFormat[0][:2]
                OriginatingIDconf = extracted_conf[0]
            except:
                OriginatingID = "Error"
                OriginatingIDconf = 0.0
            try:
                SubjectIDFormat = re.findall(r':\s?(\S+)', extracted_text[0])
                SubjectID = SubjectIDFormat[0][2:]
                SubjectIDconf = extracted_conf[0]
            except:
                SubjectID = "Error"
                SubjectIDconf = 0.0
            try:
                StudyFormat = re.findall(r':\s?(\S+)', extracted_text[0])
                Study = StudyFormat[1]
                Studyconf = extracted_conf[0]
            except:
                Study = "Error"
                Studyconf = 0.0
            try:
                DateFormat = extracted_text[-1]
                Date = DateFormat
                Dateconf = extracted_conf[-1]
            except:
                Date = "Error"
                Dateconf = 0.0
            exp_list = [OriginatingID, SubjectID, Study, Date]
            exp_conf = [OriginatingIDconf, SubjectIDconf, Studyconf, Dateconf]
            for i in range(0, len(key)):
                formated_text[key[i]] = {"Value": exp_list[i], "Conf": exp_conf[i]}
            print("label1", formated_text)
            logging.debug(formated_text)
            logging.info("extraction completed")

        elif predicted_class == "Label2":
            key = ["OriginatingID", "SubjectID", "SampleType", "Datetime"]
            formated_text = {}
            try:
                OriginatingIDFormat = extracted_text[0:2]
                OriginatingID = ' '.join(OriginatingIDFormat)
                OriginatingIDconf = extracted_conf[0]
            except:
                OriginatingID = "Error"
                OriginatingIDconf = 0.0
            try:
                SubjectIDFormat = extracted_text[1]
                SubjectID = SubjectIDFormat
                SubjectIDconf = extracted_conf[1]
            except:
                SubjectID = "Error"
                SubjectIDconf = 0.0
            try:
                SampleTypeFormat = extracted_text[2].split(" ", )
                SampleType = SampleTypeFormat[1]
                Sampletypeconf = extracted_conf[2]
            except:
                SampleType = "Error"
                Sampletypeconf = 0.0
            try:
                DatetimeFormat = extracted_text[2].split(" ", )
                Datetime = ' '.join(DatetimeFormat[2:])
                Datetimeconf = extracted_conf[2]
            except:
                Datetime = "Error"
                Datetimeconf = 0.0
            exp_list = [OriginatingID, SubjectID, SampleType, Datetime]
            exp_conf = [OriginatingIDconf, SubjectIDconf, Sampletypeconf, Datetimeconf]
            for i in range(0, len(key)):
                formated_text[key[i]] = {"Value": exp_list[i], "Conf": exp_conf[i]}
            print("label2", formated_text)
            logging.debug(formated_text)
            logging.info("extraction completed")
        elif predicted_class == "Label3":
            key = ["Date", "Site", "Study", "OriginatingID", "SampleRemark", "VisitCode", "SampleType"]
            formated_text = {}
            try:
                DateFormat = extracted_text[0]
                Date = DateFormat
                Dateconf = extracted_conf[0]
            except:
                Date = "Error"
                Dateconf = 0.0
            try:
                SiteFormat = extracted_text[3]
                Site = str(SiteFormat)[:3]
                Siteconf = extracted_conf[3]
            except:
                Site = "Error"
                Siteconf = 0.0
            try:
                StudyFormat = extracted_text[3]
                Study = str(StudyFormat)[3:]
                Studyconf = extracted_conf[3]
            except:
                Study = "Error"
                Studyconf = 0.0
            try:
                OriginatingIDFormat = extracted_text[4].split("/", )
                OriginatingID = OriginatingIDFormat[1][:4]
                OriginatingIDconf = extracted_conf[4]
            except Exception as e:
                OriginatingID = "Error"
                OriginatingIDconf = 0.0
            try:
                SampleRemarkFormat = extracted_text[5].split("/", )
                SampleRemark = SampleRemarkFormat[0]
                SampleRemarkconf = extracted_conf[5]
            except:
                SampleRemark = "Error"
                SampleRemarkconf = 0.0
            try:
                VisitCodeFormat = extracted_text[5].split("/", )
                VisitCode = VisitCodeFormat[1]
                VisitCodeconf = extracted_conf[5]
            except:
                VisitCode = "Error"
                VisitCodeconf = 0.0
            try:
                SampleTypeFormat = extracted_text[5].split("/", )
                SampleType = SampleTypeFormat[-1]
                SampleTypeconf = extracted_conf[5]
            except:
                SampleType = "Error"
                SampleTypeconf = 0.0

            exp_list = [Date, Site, Study, OriginatingID, SampleRemark, VisitCode, SampleType]
            exp_conf = [Dateconf, Siteconf, Studyconf, OriginatingIDconf, SampleRemarkconf, VisitCodeconf,
                        SampleTypeconf]
            for i in range(0, len(key)):
                formated_text[key[i]] = {"Value": exp_list[i], "Conf": exp_conf[i]}
            print("label3", formated_text)
            logging.debug(formated_text)
            logging.info("extraction completed")
        elif predicted_class == "Label5":
            key = ["OriginatingID", "Study", "SUBLOT", "PATIENT"]
            formated_text = {}
            try:
                OriginatingIDformat = extracted_text[0].split(",", 1)
                OriginatingID = OriginatingIDformat[0]
                OriginatingIDconf = extracted_conf[0]
            except:
                OriginatingID = "Error"
                OriginatingIDconf = 0.0
            try:
                Studyformat = extracted_text[0].split(",", 1)
                Study = Studyformat[1]
                Studyconf = extracted_conf[0]
            except:
                Study = "Error"
                Studyconf = 0.0
            try:
                Sublotformat = extracted_text[1].split(" ", 1)
                Sublot = Sublotformat[1]
                Sublotconf = extracted_conf[1]
            except:
                Sublot = "Error"
                Sublotconf = 0.0
            try:
                Patientformat = extracted_text[2].split(" ", 1)
                Patient = Patientformat[1]
                Patientconf = extracted_conf[2]
            except:
                Patient = "Error"
                Patientconf = 0.0
                Sublot = extracted_text[1].split(" ", 1)

            exp_list = [OriginatingID, Study, Sublot, Patient]
            exp_conf = [OriginatingIDconf, Studyconf, Sublotconf, Patientconf]
            for i in range(0, 4):
                formated_text[key[i]] = {"Value": exp_list[i], "Conf": exp_conf[i]}
            print("label5", formated_text)
            logging.debug(formated_text)
            logging.info("extraction completed")

        return {
            'Label': predicted_class,
            'Confidence': float(round_score),
            'Barcode': Barcode,
            'Handwriten': extracted_hwtext,
            'Text': extracted_text,
            'FormatedText': formated_text
        }
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

