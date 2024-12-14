import easyocr
import io
from PIL import Image
from PyPDF2 import PdfReader
import os

import shutil # optional
import cv2
import numpy as np
from pdf2image import convert_from_path
import re

pdf_file = r"ocr_material\Lich su 12.pdf"

# OCR
def readOCRfromImage(image_path):
  ocr_reader = easyocr.Reader(['vi'], gpu=True)
  text = ocr_reader.readtext(image_path, detail=0, paragraph=True)
#   return '\n '.join(text)
  return ' '.join(text)

# recall function
def fromPDFtoImg(pdf_file, page_index):
    images_folder = "image"
    if not os.path.exists(images_folder):
        create_output_folder(images_folder)
        
    text_list = []
    with open(pdf_file, 'rb') as file:
        pdf_reader = PdfReader(file)
        page = pdf_reader.pages[page_index]
        if '/XObject' in page['/Resources']:
            xobject = page['/Resources']['/XObject'].get_object()
            counter = 0
            for obj in xobject:
                if xobject[obj]['/Subtype'] == '/Image':
                    img_data = xobject[obj].get_data()
                    img_ext = xobject[obj]['/Filter']                    
                    if img_ext == '/DCTDecode':
                        img_format = 'JPEG'
                    elif img_ext == '/FlateDecode':  
                        img_format = 'PNG'
                    else:
                        print(f"Sai định dạng: {img_ext}")
                        continue
                    image = Image.open(io.BytesIO(img_data))
                    processed_image = grayscale(image)
                    processed_image.save(f"{images_folder}/extracted_image_page_{page_index + 1}_{counter}.{img_format.lower()}")
                    raw_text = readOCRfromImage(f"{images_folder}/extracted_image_page_{page_index + 1}_{counter}.{img_format.lower()}")
                    processed_text = clean_ocr_text(raw_text)
                    text_list.append(processed_text)
                    counter += 1

    return ' '.join(text_list)

# xử lí ocr 
def grayscale(image):
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
        if area > image.width * image.height * 0.1: 
            cv2.drawContours(mask, [contour], 0, 0, -1)
    result = cv2.bitwise_and(gray, mask)
    _, thresh = cv2.threshold(result, 150, 255, cv2.THRESH_BINARY_INV)
    processed_image = Image.fromarray(thresh)
    return processed_image

def clean_ocr_text(text):
    # text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)
    # Loại bỏ dòng chứa "Hình X. ..."
    text = re.sub(r'^.*?Hình\s*\d+\..*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^(\s*\d+\s*)$', '', text, flags=re.MULTILINE) # Loại bỏ số trang ở cuối dòng
    text = re.sub(r'(?m)^—\s?', '', text) # gạch đầu dòng
    # text = re.sub(r'\n+', ' ', text)# Loại bỏ nhiều dòng trắng
    # text = re.sub(r'\s+', ' ', text).strip()# Loại bỏ khoảng trắng thừa
    return text.strip()

# Tiền xử lí ocr 
def pdf_to_images_and_back(pdf_path, output_folder, output_pdf_path, dpi=300, fmt='jpeg'):
    ''' pdf => ảnh => pdf (jpeg)
    '''
    if not os.path.exists(output_folder):
        create_output_folder(output_folder)

    images = convert_from_path(pdf_path, dpi=dpi, output_folder=output_folder, fmt=fmt)

    image_paths = []
    for i, image in enumerate(images):
        image_path = f"{output_folder}/page_{i + 1}.{fmt}"
        image.save(image_path)
        image_paths.append(image_path)

    pil_images = [Image.open(image_path) for image_path in image_paths]
    pil_images = [img.convert('RGB') for img in pil_images]
    pil_images[0].save(
        output_pdf_path,
        save_all=True,
        append_images=pil_images[1:]
    )
    print(f"Lưu {output_pdf_path}")

    clean_temp_folder(image_path)

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

# Test
def main():
    # image_folder = "image"
    # create_output_folder(image_folder)
    output_folder = "output"
    create_output_folder(output_folder)

    text = ''
    for i in range(15):
        text += fromPDFtoImg(pdf_file, i)
        # text += f"\n Trang {i}\n"

    textPathOutput = f"{output_folder}/lichsu12_easy.txt"
    save_text_to_file(text,textPathOutput)

    # xóa folder ảnh
    # output_temp = "image"
    # clean_temp_folder(output_temp)

    # output_pdf_path = f"{output_folder}/lichsu12_jpeg.pdf"
    # pdf_to_images_and_back(pdf_file, output_folder, output_pdf_path, dpi=300, fmt='jpeg')

if __name__ == "__main__":
    main()
## can xoa
# ten hinh ben duoi
# cac cau hoi cham hoi
# Cau hoi va tra loi