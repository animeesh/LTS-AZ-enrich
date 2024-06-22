import boto3
import re
from PIL import Image
from cvu.detector.yolov5 import Yolov5
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import tensorflow as tf
from pyzbar.pyzbar import decode
import var
from datetime import date

IC_CLASS_NAMES = ["sanofi_001", "sanofi_002", "sanofi_003", "sanofi_004", "sanofi_005","sanofi_007"]

TUBE_OD_CLASS_NAMES = ["Label", "Tube", "base"]  # Ml model for crop and predict

TUBE_MODEL_OD = Yolov5(classes=TUBE_OD_CLASS_NAMES, backend="onnx", weight="models/sanofiModel/tube/110424.onnx", device="cpu")

MODEL_IC_crop = tf.keras.models.load_model("models/sanofiModel/IC/sanofi1_0.h5")
AWS_TEXTRACT = boto3.client('textract', aws_access_key_id=var.aws_access_key_id,
                            aws_secret_access_key=var.aws_secret_access_key,
                            region_name=var.region)



async def predictSanofi(
        logging, 
        file: UploadFile = File(...)):
    imageapi = await file.read()
    imageapiOD = Image.open(BytesIO(imageapi))
    image = np.array(imageapiOD.convert('RGB'))
    predictions_OD = TUBE_MODEL_OD(image)  # results of OD

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


    try:
        # for label image
        max_dict = max(det_list, key=lambda x: x["size"])
        max_size = max_dict["size"]
        max_image = max_dict["im"]
        best_image = np.array(max_image)
        image1 = Image.fromarray(best_image)

        # image1.show()
        # filename = f"image_time.png"
        # image1.save(filename)
        max_class = max_dict["class"]

        # crop classification

        # image_resized_crop = cv2.resize(best_image, (512, 512))
        image_resized_crop = image1.resize((512, 512))
        image_dims_crop = np.expand_dims(image_resized_crop, axis=0)
        predictions_crop = MODEL_IC_crop.predict(image_dims_crop)
        print(predictions_crop)
        score_crop = (predictions_crop[0])
        score_crop = (100 * np.max(score_crop))
        round_crop_score = round(score_crop, 2)
        predicted_class = IC_CLASS_NAMES[np.argmax(predictions_crop[0])]
        print("this is crop clasifiaction")
        print(predicted_class, round_crop_score)

        # crop end code
        print("actual class :", max_class)
        # if max_class == "Labels_IN":
        #     image1 = image1.rotate(180)  # rotating
        #     print("image is been rotated upside-down")
        # if predicted_class == "Label3":
        #     image1 = image1.rotate(-90, expand=True)  # rotating
        #     print("image is been rotated clockwise")
        # if predicted_class == "Label5":
        #     image1 = image1.rotate(90, expand=True)  # rotating
        #     print("image is been rotated anticlockwise")

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
        formated_text = {}
        if predicted_class == "sanofi_001":
            sanofi001(formated_text, extracted_text,extracted_conf)
            print("sanofi_001", formated_text)
        elif predicted_class == "sanofi_002":
            key = ["OriginatingID", "Study", "Site", "SubjectID", "Visit"]
            str = ' '.join(extracted_text)
            try:
                OriginatingIDFormat = re.search(r"Subject\s*:\s*(\w+)", str)
                OriginatingID = OriginatingIDFormat.group(1)
                OriginatingIDconf = extracted_conf[0]
            except:
                OriginatingID = "Error"
                OriginatingIDconf = 0.0
            try:
                StudyFormat = re.search(r"Study\s*:\s*(\w+)", str)
                Study = StudyFormat.group(1)
                Studyconf = extracted_conf[1]
            except:
                Study = "Error"
                Studyconf = 0.0
            try:
                SiteFormat = re.search(r"Site\s*:\s*(\w+)", str)
                Site = SiteFormat.group(1)
                Siteconf = extracted_conf[2]
            except:
                Site = "Error"
                Siteconf = 0.0
            try:
                SubjectIDFormat = re.search(r"Subject\s*:\s*(\w+)", str)
                SubjectID = SubjectIDFormat.group(1)
                SubjectIDconf = extracted_conf[0]
            except:
                SubjectID = "Error"
                SubjectIDconf = 0.0
            try:
                VisitFormat = re.search(r"Sample Type\s*:\s*(\w+)", str)
                Visit = VisitFormat.group(1)
                Visitconf = extracted_conf[-1]
            except:
                Visit = "Error"
                Visitconf = 0.0
            exp_list = [OriginatingID, Study, Site, SubjectID, Visit]
            exp_conf = [OriginatingIDconf, Studyconf, Siteconf, SubjectIDconf, Visitconf]
            for i in range(0, len(key)):
                formated_text[key[i]] = {"Value": exp_list[i], "Conf": exp_conf[i]}
            print("sanofi_002", formated_text)

        elif predicted_class == "sanofi_003":
            key = ["OriginatingID", "Study", "Site" ,"SubjectID", "Visit"]
            str = ' '.join(extracted_text)
            try:
                OriginatingIDFormat = re.search(r"Subject\s*:\s*(\w+)", str)
                OriginatingID = OriginatingIDFormat.group(1)
                OriginatingIDconf = extracted_conf[0]
            except:
                OriginatingID = "Error"
                OriginatingIDconf = 0.0
            try:
                StudyFormat =  re.search(r"Study\s*:\s*(\w+)", str)
                Study = StudyFormat.group(1)
                Studyconf = extracted_conf[1]
            except:
                Study = "Error"
                Studyconf = 0.0
            try:
                SiteFormat =  re.search(r"Site\s*:\s*(\w+)", str)
                Site = SiteFormat.group(1)
                Siteconf = extracted_conf[2]
            except:
                Site = "Error"
                Siteconf = 0.0
            try:
                SubjectIDFormat = re.search(r"Subject\s*:\s*(\w+)", str)
                SubjectID = SubjectIDFormat.group(1)
                SubjectIDconf = extracted_conf[0]
            except:
                SubjectID = "Error"
                SubjectIDconf = 0.0
            try:
                VisitFormat =  re.search(r"Sample Type\s*:\s*(\w+)", str)
                Visit = VisitFormat.group(1)
                Visitconf = extracted_conf[-1]
            except:
                Visit = "Error"
                Visitconf = 0.0
            exp_list = [OriginatingID, Study, Site ,SubjectID, Visit]
            exp_conf = [OriginatingIDconf, Studyconf, Siteconf,SubjectIDconf, Visitconf]
            for i in range(0, len(key)):
                formated_text[key[i]] = {"Value": exp_list[i], "Conf": exp_conf[i]}
            print("sanofi_003", formated_text)

        elif predicted_class == "sanofi_004":
            key = ["OriginatingID", "Study","SubjectID","SampleRemark", "Date"]
            str = ' '.join(extracted_text)

            try:
                OriginatingIDFormat = re.search(r"ID#\s*(\w+)", str)
                OriginatingID = OriginatingIDFormat.group(1)
                OriginatingIDconf = extracted_conf[0]
            except:
                OriginatingID = "Error"
                OriginatingIDconf = 0.0
            try:
                StudyFormat = extracted_text[0]
                Study = StudyFormat
                Studyconf = extracted_conf[1]
            except:
                Study = "Error"
                Studyconf = 0.0
            try:
                SubjectIDFormat = re.search(r"ID#\s*(\w+)", str)
                SubjectID = SubjectIDFormat.group(1)
                SubjectIDconf = extracted_conf[2]
            except:
                SubjectID = "Error"
                SubjectIDconf = 0.0
            try:
                SampleRemarkFormat = extracted_text[1]
                SampleRemark = SampleRemarkFormat
                SampleRemarkconf = extracted_conf[4]
            except:
                SampleRemark = "Error"
                SampleRemarkconf = 0.0
            try:
                DateFormat = re.search(r"Date\s*(\d{2}/\d{2}/\d{2})", str)
                Date = DateFormat.group(1)
                Dateconf = extracted_conf[-1]
            except:
                Date = "Error"
                Dateconf = 0.0
            exp_list = [OriginatingID, Study,  SubjectID,SampleRemark, Date]
            exp_conf = [OriginatingIDconf, Studyconf, SubjectIDconf, SampleRemarkconf, Dateconf]
            for i in range(0, len(key)):
                formated_text[key[i]] = {"Value": exp_list[i], "Conf": exp_conf[i]}
            print("sanofi_004", formated_text)

        elif predicted_class == "sanofi_005":
            sanofi005(formated_text, extracted_text,extracted_conf)
            print("sanofi_005", formated_text)

        elif predicted_class == "sanofi_007":
            key = ["OriginatingID", "Study", "Site", "SubjectID", "Visit","SampleRemark"]
            str = ' '.join(extracted_text)
            try:
                OriginatingIDFormat = re.search(r"Study\s*:\s*(\w+)", str)  # str[str.index(()+len()):str.index("")].strip()#extracted_text[6]
                OriginatingID = OriginatingIDFormat.group(1)
                OriginatingIDconf = extracted_conf[0]
            except:
                OriginatingID = "Error"
                OriginatingIDconf = 0.0
            try:
                StudyFormat = re.search(r"Study\s*:\s*(\w+)", str)
                Study = StudyFormat.group(1)
                Studyconf = extracted_conf[1]
            except:
                Study = "Error"
                Studyconf = 0.0
            try:
                SiteFormat = re.search(r"Study\s*:\s*(\w+)", str)
                Site = SiteFormat.group(1)
                Siteconf = extracted_conf[2]
            except:
                Site = "Error"
                Siteconf = 0.0
            try:
                SubjectID = OriginatingID
                #SubjectID = SubjectIDFormat
                SubjectIDconf = extracted_conf[6]
            except:
                SubjectID = "Error"
                SubjectIDconf = 0.0
            try:
                VisitFormat = re.search(r"Study\s*:\s*(\w+)", str)
                Visit = VisitFormat.group(1)
                Visitconf = extracted_conf[-1]
            except:
                Visit = "Error"
                Visitconf = 0.0
            try:
                SampleRemarkFormat = re.search(r"Study\s*:\s*(\w+)", str)
                SampleRemark = SampleRemarkFormat.group(1)
                SampleRemarkconf = extracted_conf[-1]
            except:
                SampleRemark = "Error"
                SampleRemarkconf = 0.0
            exp_list = [OriginatingID, Study, Site, SubjectID, Visit,SampleRemark]
            exp_conf = [OriginatingIDconf, Studyconf, Siteconf, SubjectIDconf, Visitconf,SampleRemarkconf]
            for i in range(0, len(key)):
                formated_text[key[i]] = {"Value": exp_list[i], "Conf": exp_conf[i]}
            print("sanofi_006", formated_text)

        logging.debug(formated_text)
        logging.info("extraction completed")

        return {
            'Label': predicted_class,
            'Confidence': float(round_crop_score),
            'Barcode': Barcode,
            'Handwriten': ["no-handwritten"],
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

def sanofi001(formated_text, extracted_text,extracted_conf):
    key = ["OriginatingID", "Study", "Site", "SubjectID", "Visit"]
    str = ' '.join(extracted_text)
    try:
        OriginatingIDFormat = re.search(r"Subject\s*:\s*(\w+)", str)
        OriginatingID = OriginatingIDFormat.group(1)
        OriginatingIDconf = extracted_conf[0]
    except:
        OriginatingID = "Error"
        OriginatingIDconf = 0.0
    try:
        StudyFormat = re.search(r"Study\s*:\s*(\w+)", str)
        Study = StudyFormat.group(1)
        Studyconf = extracted_conf[1]
    except:
        Study = "Error"
        Studyconf = 0.0
    try:
        SiteFormat = re.search(r"Site\s*:\s*(\w+)", str)
        Site = SiteFormat.group(1)
        Siteconf = extracted_conf[2]
    except:
        Site = "Error"
        Siteconf = 0.0
    try:
        SubjectIDFormat = re.search(r"Subject\s*:\s*(\w+)", str)
        SubjectID = SubjectIDFormat.group(1)
        SubjectIDconf = extracted_conf[0]
    except:
        SubjectID = "Error"
        SubjectIDconf = 0.0
    try:
        VisitFormat = re.search(r"Visit\s*:\s*(\w+)", str)
        Visit = VisitFormat.group(1)
        Visitconf = extracted_conf[-1]
    except:
        Visit = "Error"
        Visitconf = 0.0
    exp_list = [OriginatingID, Study, Site, SubjectID, Visit]
    exp_conf = [OriginatingIDconf, Studyconf, Siteconf, SubjectIDconf, Visitconf]
    setText(formated_text, key,exp_list,exp_conf)


def sanofi005(formated_text, extracted_text,extracted_conf):
    key = ["OriginatingID", "Study", "Site", "SubjectID", "Visit"]
    str = ' '.join(extracted_text)
    exp_list = []
    exp_conf = []
    extract(str, r"Subject\s*:\s*(\w+)",exp_list, exp_conf, extracted_conf, 0)
    extract(str, r"Study\s*:\s*(\w+)",exp_list, exp_conf, extracted_conf, 1)
    extract(str, r"Site\s*:\s*(\w+)",exp_list, exp_conf, extracted_conf, 2)
    extract(str, r"Subject\s*:\s*(\w+)",exp_list, exp_conf, extracted_conf, 0)
    extract(str, r"Sample type\s*:\s*(\w+)",exp_list, exp_conf, extracted_conf, -1)
    setText(formated_text, key,exp_list,exp_conf)
    
def setText(formated_text, key,exp_list,exp_conf):
    for i in range(0,len(key)):
        formated_text[key[i]] = {"Value": exp_list[i], "Conf": exp_conf[i]}
        
def extract(str, regex,exp_list, exp_conf, extracted_conf, index):    
    try:
        result = re.search(regex, str)
        exp_list.append( result.group(1))
        exp_conf.append(extracted_conf[index])
    except:
        exp_list.append("Error")
        exp_conf.append(0.0)