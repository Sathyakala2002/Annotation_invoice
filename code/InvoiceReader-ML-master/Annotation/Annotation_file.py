import os
import cv2
import pytesseract
import re
import json

# Path to Tesseract OCR executable (change accordingly)
# pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def extract_text(image_path):
    image = cv2.imread(image_path)
   
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   
    inverted_image = cv2.bitwise_not(thresh_image)
   
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_image = cv2.dilate(inverted_image, kernel, iterations=1)
   
    # Perform OCR
    text = pytesseract.image_to_string(dilated_image)
    print(text,"text")
    return text


def extract_key_value_pairs_and_line_items(text):
    key_value_pairs = {}
    line_items = []

    lines = text.split('\n')
    key_value_pattern = re.compile(r'([^:]+)\s*:\s*(.*)')

    # Define regular expression pattern for line items
    line_item_pattern = r'(\d+)\s+(.*?)\s+(\d+\.\d+)\s+(\d+\.\d+)'

    for line in lines:
        key_value_match = key_value_pattern.match(line)
        if key_value_match:
            key = key_value_match.group(1).strip()
            value = key_value_match.group(2).strip()
            key_value_pairs[key] = value
        else:
            line_item_match = re.match(line_item_pattern, line)
            if line_item_match:
                quantity = int(line_item_match.group(1))
                description = line_item_match.group(2).strip()
                unit_price = float(line_item_match.group(3))
                total_price = float(line_item_match.group(4))

                line_item = {
                    "quantity": quantity,
                    "description": description,
                    "unit_price": unit_price,
                    "total_price": total_price
                }
                line_items.append(line_item)

    return key_value_pairs, line_items

def save_as_json(key_value_pairs, line_items, output_file):
    data = {
        "key_value_pairs": key_value_pairs,
        "line_items": line_items
    }
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def process_images_in_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_directory, filename)
            output_json = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.json")

            text = extract_text(image_path)

            key_value_pairs, line_items = extract_key_value_pairs_and_line_items(text)

            save_as_json(key_value_pairs, line_items, output_json)

            print("Annotations saved as JSON:", output_json)

input_directory = 'C:/Users/admin/Documents/imagetotext- python/invoicemodel/Invoice/InvoiceReader-ML-master/templates'  # Replace with your input directory containing invoice images
output_directory = 'InvoiceReader-ML-master/Output_Annotation'  # Replace with your output directory for JSON annotations

process_images_in_directory(input_directory, output_directory)
