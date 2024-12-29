import base64
import requests
from pdf2image import convert_from_path
import os
from PyPDF2 import PdfReader
ngrok_url ="https://2dd4-34-125-213-13.ngrok-free.app/"
# Usage
def upload_image_to_colab(image_path):
    with open(image_path, "rb") as image_file:
        response = requests.post(
            url=f"{ngrok_url}/upload",  # URL Ngrok của bạn
            files={"file": image_file}  # Gửi file qua form-data
        )
    if response.status_code == 200:
        print("Upload successful:", response.json())
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# def perform_ocr(image_path):
#     # Bước 1: Upload ảnh lên Colab
#     upload_response = upload_image_to_colab(image_path)
#     if not upload_response or "file_path" not in upload_response:
#         print("Error: Failed to upload image.")
#         return ""

#     # Lấy đường dẫn ảnh đã upload
#     colab_image_path = upload_response["file_path"]
#     print(colab_image_path)
#     # Bước 2: Gọi API OCR trên Colab
#     response = requests.post(
#         url=f"{ngrok_url}/ocr",
#         json={"image_path": colab_image_path},  # Truyền đường dẫn ảnh trên Colab
#     )
#     if response.status_code == 200:
#         return response.json().get("response_message", "")
#     else:
#         print(f"Error: {response.status_code}, {response.text}")
#         return ""
        
def perform_ocr(image_path, images_folder, page_index):
    path_image = upload_image_to_colab(image_path)
    image_path = f"{images_folder}/page_{page_index}.png"
    print(image_path)
    print("path image", path_image)
    colab_image_path = path_image["file_path"]
    print("COLAB", colab_image_path)
    response = requests.post(
        url=f"{ngrok_url}/ocr", # Ensure this URL matches your Ollama service endpoint
        json={
            "image_url": image_path,
        },
    )
    print("Response in =", response.elapsed.total_seconds())
    if response.status_code == 200:
        return response.json().get("response_message")
    else:
        print("Error:", response.status_code, response.text)
        return None

# Function
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

def fromPDFtoImg(pdf_file, page_index):
    images_folder = "images"
    if not os.path.exists(images_folder):
        create_output_folder(images_folder)

    # Chuyển trang PDF thành ảnh
    images = convert_pdf_to_images(pdf_file, images_folder, first_page=page_index + 1, last_page=page_index + 1)
    if not images:
        return ""

    # Xử lý từng ảnh
    for idx, image in enumerate(images):
        # Lưu ảnh vào tệp PNG
        image_path = os.path.join(images_folder, f"page_{page_index}.png")
        image.save(image_path, format="PNG")

        # Gọi OCR sau khi upload ảnh
        result = perform_ocr(image_path, images_folder, page_index)
        with open("lichsu.txt", "a", encoding="utf-8") as file:
            file.write(f"Page {page_index + idx}:\n{result}\n")

def create_output_folder(folder_path):
    """Tạo thư mục lưu trữ nếu chưa tồn tại."""
    os.makedirs(folder_path, exist_ok=True)
    print(f"Tạo '{folder_path}'")

##################### TEST ##############################
# image_path = "https://img.otofun.net/upload/v7/images/6502/6502555-b4ab70a6be8c509885ca7ecb996bcc71.jpg"  # Replace with your image path

image_path = "images/page_10.png"
# image_path = "lichsu_test_local.png"

# Show image
# im = Image.open(image_path)
# im.show()

# result = perform_ocr(image_path)

# print(result)
# output_file="lichsu.txt"
# with open(output_file, "w", encoding="utf-8") as file:
#   file.write(result)
pdfPath = "sachSu_12.pdf"
reader = PdfReader(pdfPath)
for i in range(0, len(reader.pages)):
    fromPDFtoImg(pdfPath, i)
# fromPDFtoImg(pdfPath, 12)