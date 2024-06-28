
import cv2
from pyzbar.pyzbar import decode
import constant
from io import BytesIO
import numpy as np
from PIL import Image
import os

inputfolder= "generic/sanofi_015" #"generic/"
#outputfolder = "output/"
#outputfolder ="croppedHR"
outputfolder = "croppedtext/news15"
#outputfolder = "bounding_output/"

def imageODtest(inputfolder,outputfolder):
    files = os.listdir(inputfolder)
    for file_name in files:
        input_path= os.path.join(inputfolder,file_name)
        image = cv2.imread(input_path)
        print(file_name)
        predictions = constant.GenericODModel(image)  # results of OD
        det_list = []

        # loop through predictions
        for prediction in predictions:
            det_dict = {}
            x1 = int(prediction.bbox[0])
            y1 = int(prediction.bbox[1])
            x2 = int(prediction.bbox[2])
            y2 = int(prediction.bbox[3])
            cropped = image[y1:y2, x1:x2]

            if prediction.class_name == "Barcode":
                detectedBarcodes = decode(cropped)
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
                pass

            if prediction.class_name == "HR":
                best_image = np.array(cropped)
                image1 = Image.fromarray(best_image)
                imageHR= image1.rotate(-90,expand=True)
                imgByteArr = BytesIO()
                imageHR.save(imgByteArr, format='JPEG')
                imgByteArr.seek(0)
                extraction = constant.AWS_TEXTRACT.detect_document_text(Document={'Bytes': imgByteArr.getvalue()})
                extracted_HR = []
                extracted_HRconf = []
                for item in extraction['Blocks']:
                    if item['BlockType'] == 'LINE':
                        # extracted_text += item['Text'] + '\n'
                        extracted_HR += item['Text'],
                        extracted_HRconf += round(item['Confidence'], 2),

                print("extracted_HR = ", extracted_HR)
                # outputpath = os.path.join(outputfolder, file_name)
                # rotated_cropped = cv2.rotate(cropped,cv2.ROTATE_90_CLOCKWISE)
                # cv2.imwrite(outputpath, rotated_cropped)

            if prediction.class_name == "Text":
                Text_image = np.array(cropped)
                Text_image_array = Image.fromarray(Text_image)
                TextByteArr = BytesIO()
                Text_image_array.save(TextByteArr, format='JPEG')
                TextByteArr.seek(0)
                extraction= constant.AWS_TEXTRACT.detect_document_text(Document={'Bytes': TextByteArr.getvalue()})
                extracted_Text =[]
                extracted_Textconf =[]
                for item in extraction['Blocks']:
                    if item['BlockType'] == 'LINE':
                    # extracted_text += item['Text'] + '\n'
                        extracted_Text += item['Text'],
                        extracted_Textconf += round(item['Confidence'], 2),

                print("extracted_text = ",extracted_Text)
                outputpath = os.path.join(outputfolder, file_name)
                cv2.imwrite(outputpath, cropped)

            if prediction.class_name == "Noise":
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
                print("Noise removed")

            height, width, _ = cropped.shape
            size = width * height
            det_dict["class"] = prediction.class_name
            #det_dict["im"] = cropped
            det_dict["size"] = size
            det_list.append(det_dict)

            #prediction.draw(image)

        print(det_list)
        # outputpath = os.path.join(outputfolder,file_name)
        # cv2.imwrite(outputpath, image)

imageODtest(inputfolder,outputfolder)




