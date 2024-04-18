import cv2
import pytesseract
import numpy as np

image_path = 'C:/Users/admin/Documents/Aroopa tasks/imagetotext- python/invoicemodel/Invoice/InvoiceReader-ML-master/image2.jpg'

image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

word_data = pytesseract.image_to_data(gray_image, output_type=pytesseract.Output.DICT)

def extract_text(img, x, y, w, h):
    crop_img = img[y:y + h, x:x + w]
    text = pytesseract.image_to_string(crop_img)
    cleaned_text = ' '.join(text.split())
    return cleaned_text

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

# Sort contours based on their y-coordinate
contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[1])

# Extract text for each contour and save to a text file
output_file = 'extracted_text_with_position.txt'
with open(output_file, 'w') as f:
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        text = extract_text(image, x, y, w, h)
        # Adjust y-coordinate to match the origin at the bottom-left corner of the image
        y = gray_image.shape[0] - y
        f.write(f"{text.strip()} ({x},{y}) ({x + w},{y + h})\n")

print(f"Extracted text with positions saved to {output_file}")
