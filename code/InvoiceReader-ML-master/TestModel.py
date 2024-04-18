import cv2
import pytesseract
import spacy
import numpy as np
import json
import os

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def extract_text(img, x, y, w, h):
    crop_img = img[y:y + h, x:x + w]
    text = pytesseract.image_to_string(crop_img)
    cleaned_text = ' '.join(text.split())
    return cleaned_text

def extract_and_store_entities(image, entities, nlp):
    x1 = 50
    y1 = 10
    imgo = image.copy()
    gray = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
    linek = np.zeros((11, 11), dtype=np.uint8)
    linek[5, ...] = 1
    x = cv2.morphologyEx(gray, cv2.MORPH_OPEN, linek, iterations=1)
    gray -= x
    gray = cv2.bitwise_not(gray)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    while True:
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (x1, y1))
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=6)
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        test = len(contours)

        if test < 10:
            x1 -= 1
            y1 -= 1
        elif test > 30:
            x1 += 1
            y1 += 1
        else:
            break

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        text = extract_text(imgo, x, y, w, h)
        doc = nlp(text)
        for ent in doc.ents:
            entities.append({"label": ent.label_, "entity": ent.text})

    output_directory = "C:/Users/admin/Documents/imagetotext- python/invoicemodel/Invoice/InvoiceReader-ML-master/Output_JSON"
    output_filename = "invoice_json.json"
    
    os.makedirs(output_directory, exist_ok=True)
    
    output_json_file = os.path.join(output_directory, output_filename)
    invoice_data = {}
    invoice_data["Invoice"] = {"entities": entities}
    
    with open(output_json_file, "w") as json_file:
        json.dump(invoice_data, json_file, indent=4)
    
    print("JSON file saved at:", output_json_file)  

def process_single_invoice_image(image_path):
    nlp = spacy.load("ner_model")
    entities = []  
    image = cv2.imread(image_path)
    extract_and_store_entities(image, entities, nlp)

image_path = "C:/Users/admin/Documents/imagetotext- python/invoicemodel/Invoice/InvoiceReader-ML-master/image.jpg"
process_single_invoice_image(image_path)
