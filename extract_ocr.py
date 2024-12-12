import easyocr
import io
from PIL import Image
from PyPDF2 import PdfReader

pdf_file = "Lich su 12.pdf"

def readOCRfromImage(image_path):
  ocr_reader = easyocr.Reader(['vi'], gpu=True)
  text = ocr_reader.readtext(image_path, detail=0, paragraph=True)
  return ' '.join(text)
  

  
def fromPDFtoImg(pdf_file, page_index):
  text_list = []
  with open(pdf_file, 'rb') as file:
    pdf_reader = PdfReader(file)
    page = pdf_reader.pages[page_index]
    if ('/XObject' in page['/Resources']):
      xobject = page['/Resources']['/XObject'].get_object()
      counter = 0
      for obj in xobject:
        if xobject[obj]['/Subtype'] == '/Image':
          img_data = xobject[obj].get_data()
          img_ext = xobject[obj]['/Filter']
          if (img_ext == '/DCTDecode'):
            img_format = 'JPEG'
          elif (img_ext == '/FlateDecode'):
            img_format = 'PNG'
          else: 
            continue
          image =  Image.open(io.BytesIO(img_data))
          image.resize((image.size[0] * 2, image.size[1] * 2))
          img_name = f"image/extracted_image_page_{page_index + 1}_{counter}.{img_format.lower()}"
          image.save(img_name)
          text_list.append(readOCRfromImage(img_name))
          counter += 1
  return ' '.join(text_list)
