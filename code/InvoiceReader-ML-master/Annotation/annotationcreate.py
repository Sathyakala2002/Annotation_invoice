import os
import cv2
import pytesseract
import json
import spacy
 
nlp = spacy.load("en_core_web_sm")
 
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
 
# Function to extract text from image using OCR
def extract_text(image_path):
    image = cv2.imread(image_path)
   
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    # Perform thresholding
    _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   
    # Invert image
    inverted_image = cv2.bitwise_not(thresh_image)
   
    # Perform dilation to enhance text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_image = cv2.dilate(inverted_image, kernel, iterations=1)
   
    # Perform OCR
    text = pytesseract.image_to_string(dilated_image)
   
    return text
 
# Function to extract key-value pairs from text using spaCy NER
def extract_key_value_pairs(text):
    key_value_pairs = {}
   
    # Process text with spaCy NER
    doc = nlp(text)
   
    # Extract entities and their labels
    for ent in doc.ents:
        # Use entity text as key and entity label as value
        key_value_pairs[ent.text.strip()] = ent.label_
   
    return key_value_pairs
 
# Function to save key-value pairs as JSON
def save_as_json(key_value_pairs, output_file):
    with open(output_file, 'w') as f:
        json.dump(key_value_pairs, f, indent=4)
 
# Process all images in a directory
def process_images_in_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
 
    for filename in os.listdir(input_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_directory, filename)
            output_json = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.json")
 
            # Extract text from image
            text = extract_text(image_path)
 
            # Extract key-value pairs from text using spaCy NER
            key_value_pairs = extract_key_value_pairs(text)
 
            # Save key-value pairs as JSON
            save_as_json(key_value_pairs, output_json)
 
            print("Annotations saved as JSON:", output_json)
 
# Example usage
input_directory = 'D:/Ranjith/work/invoice/spacy_model/InvoiceReader-ML-master/InvoiceReader-ML/templates'
output_directory = 'D:/Ranjith/work/invoice/annotation'  
 
process_images_in_directory(input_directory, output_directory)