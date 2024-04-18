import spacy
from spacy.training import Example
import random
import cv2
import pytesseract
import numpy as np
import json
import os

# Load the NER model
nlp = spacy.load("ner_model")

def extract_text(img, x, y, w, h):
    crop_img = img[y:y + h, x:x + w]
    text = pytesseract.image_to_string(crop_img)
    cleaned_text = ' '.join(text.split())
    return cleaned_text

def convert_to_examples(manual_annotations):
    examples = []
    for annotation in manual_annotations:
        text = annotation["content"]
        entities = annotation["annotation"]["entities"]
        example_entities = [(start, end, label) for start, end, label in entities]
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, {"entities": example_entities})
        examples.append(example)
    return examples

annotation_file = "C:/Users/admin/Documents/Aroop_tasks/imagetotext- python/invoicemodel/Invoice/InvoiceReader-ML-master/Training_data/training_data.json"


# List to store manual annotations
manual_annotations = []

def convert_to_examples(manual_annotations):
    examples = []
    for annotation in manual_annotations:
        text = annotation["content"]
        entities = annotation["annotation"]["entities"]
        example_entities = [(start, end, label) for start, end, label in entities]
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, {"entities": example_entities})
        examples.append(example)
    return examples

# Convert manual annotations to spaCy Example objects
examples = convert_to_examples(manual_annotations)

random.shuffle(examples)

# Get the NER pipe
ner = nlp.get_pipe("ner")

# Add labels from manual annotations
for annotation in manual_annotations:
    for ent in annotation.get("annotation", {}).get("entities", []):
        ner.add_label(ent[2])

optimizer = nlp.begin_training()

# Train the model on manual annotations
for i in range(10): 
    random.shuffle(examples)
    for example in examples:
        nlp.update([example], sgd=optimizer)

# Path to the invoice image file
image_path = 'C:/Users/admin/Documents/Aroop_tasks/imagetotext- python/invoicemodel/Invoice/InvoiceReader-ML-master/image2.jpg'

# Read the image using OpenCV
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

x1 = 40
y1 = 12
imgo = gray_image.copy()  
linek = np.zeros((11, 11), dtype=np.uint8)
linek[5, ...] = 1
x = cv2.morphologyEx(imgo, cv2.MORPH_OPEN, linek, iterations=1)
gray = imgo - x
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

contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[1])

# List to store automatic annotations
automatic_annotations = []

# Extract text and positions for each contour
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    text = extract_text(image, x, y, w, h)
    y = gray_image.shape[0] - y
    annotation = {"content": text, "annotation": {"entities": [[x, x+w]]}}
    automatic_annotations.append(annotation)

# Save automatic annotations to a separate JSON file
output_file = "automatic_annotations_file.json"
with open(output_file, 'w') as file:
    json.dump(automatic_annotations, file)

# Combined manual and automatic annotations
combined_annotations = manual_annotations + automatic_annotations

# Convert combined annotations to spaCy Example objects
combined_examples = convert_to_examples(combined_annotations)

# Retrain the model with the combined dataset
for i in range(10):  
    random.shuffle(combined_examples)
    for example in combined_examples:
        nlp.update([example], sgd=optimizer)
