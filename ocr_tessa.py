from pdf2image import convert_from_path
import pytesseract
import os
import shutil 
import re
import cv2
import numpy as np
from PIL import Image
import torch

# OCR
def clean_ocr_text(text):
    # text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)
    # Loại bỏ dòng chứa "Hình X. ..."
    text = re.sub(r'^.*?Hình\s*\d+\..*$', '', text, flags=re.MULTILINE)
    # Loại bỏ số trang ở cuối dòng
    text = re.sub(r'^(\s*\d+\s*)$', '', text, flags=re.MULTILINE)
    # gạch đầu dòng
    text = re.sub(r'(?m)^—\s?', '', text)
    # # Loại bỏ nhiều dòng trắng
    # text = re.sub(r'\n+', ' ', text)
    # # Loại bỏ khoảng trắng thừa
    # text = re.sub(r'\s+', ' ', text).strip()
    return text.strip()

def convert_pdf_to_images(pdf_path, output_folder, first_page=None, last_page=None, dpi=300):
    """(str, str, int, int, int) => list
    Chuyển đổi file PDF thành danh sách các ảnh với phạm vi trang tuỳ chọn.
    """
    images = convert_from_path(
        pdf_path, 
        dpi=dpi, 
        output_folder=output_folder, 
        fmt='jpeg',
        first_page=first_page,  
        last_page=last_page   
    )
    return images

def preprocess_image(image):
    """(pil.Image) => pil.Image
    Tiền xử lý ảnh, chuyển anh sang thư viện openCV, 
    đổi sang ảnh xám và phát hiện các vùng là điểm ảnh=> tìm đường viền
    """
    open_cv_image = np.array(image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(gray.shape, dtype=np.uint8) * 255
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > image.width * image.height * 0.1:  # loại bỏ các vùng chiếm hơn 10% diện tích trang
            cv2.drawContours(mask, [contour], 0, 0, -1)    
    result = cv2.bitwise_and(gray, mask)    
    _, thresh = cv2.threshold(result, 150, 255, cv2.THRESH_BINARY_INV)    
    processed_image = Image.fromarray(thresh)
    return processed_image

def perform_ocr_on_images(images, lang="vie"):
    """(list, str) => str
    Thực hiện OCR trên danh sách các ảnh đã xử lý.
    """
    text = ""
    for i, image in enumerate(images):
        print(f"Current page : {i + 1}/{len(images)}")
        
        processed_image = preprocess_image(image)
        ocr_text = pytesseract.image_to_string(processed_image, lang=lang)
        text += ocr_text
        text = clean_ocr_text(text)
    return text

# process file
def create_output_folder(folder_path):
    """Tạo thư mục lưu trữ nếu chưa tồn tại."""
    os.makedirs(folder_path, exist_ok=True)
    print(f"Tạo '{folder_path}'")

def save_text_to_file(text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Lưu {output_path}")

def clean_temp_folder(folder_path):
    shutil.rmtree(folder_path)
    print(f"Xóa '{folder_path}'")

# recall function
def fromPDFtoImg(pdf_file, page_index):
    """
    ocr ảnh
    """
    images_folder = "image"
    if not os.path.exists(images_folder):
      create_output_folder(images_folder)
    images = convert_pdf_to_images(pdf_file, images_folder, first_page=page_index + 1, last_page=page_index + 1)
    if not images:
        return ""  
    processed_image = preprocess_image(images[0])
    ocr_text = pytesseract.image_to_string(processed_image, lang="vie")
    return ''.join(clean_ocr_text(ocr_text))
    # return clean_ocr_text(ocr_text)
# path 
# ocr Tessaract

# # Test
# def main():
#     pdf_file = r"ocr_material/Lich su 12.pdf"
#     # image_folder = "image"
#     # create_output_folder(image_folder)
#     output_folder = "output"
#     create_output_folder(output_folder)

#     text = ''
#     for i in range(15):
#       text += fromPDFtoImg(pdf_file, i)
#       # text += f"\n Trang {i}\n"

#     textPathOutput = f"{output_folder}//lichsu12_tessa.txt"
#     save_text_to_file(text,textPathOutput)

#     # xóa folder ảnh
#     # output_temp = "image"
#     # clean_temp_folder(output_temp)

# if __name__ == "__main__":
#     main()
