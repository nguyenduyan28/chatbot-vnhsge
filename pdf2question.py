import torch
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, pipeline
import re
import os
import json
import easyocr
from ocr_tessa import fromPDFtoImg
from verify import verify_qa
from test_sentence import semantic_chunk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from underthesea import sent_tokenize

# Set device
device = torch.device("mps")
model = HuggingFaceEmbeddings(model_name='dangvantuan/vietnamese-embedding')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

material_path = "ocr_material"
pdf_name = "Dia Ly 12 canh dieu.pdf"
file_name = os.path.splitext(pdf_name)[0].replace(' ', '_')
print(file_name)
os.makedirs(os.path.join('qa_pairs_ocr', file_name), exist_ok=True)
pdf_file = os.path.join(material_path, pdf_name)
output_file = f"qa_pairs_ocr/{file_name}/{file_name}_qa_pairs.json"
open(output_file, 'w').close()

def chunk_to_sub_paragraph(text, chunk_size=300):
    sentences = sent_tokenize(text)
    buffer = ''
    chunk = []
    for sentence in sentences:
        if len(buffer) >= chunk_size:
            chunk.append(buffer)
            buffer = ''
        buffer += sentence
    return chunk

# Load models and pipelines
query_model_name = "doc2query/msmarco-vietnamese-mt5-base-v1"
query_tokenizer = AutoTokenizer.from_pretrained(query_model_name)
query_model = AutoModelForSeq2SeqLM.from_pretrained(query_model_name).to(device)

qa_tokenizer = AutoTokenizer.from_pretrained("ancs21/xlm-roberta-large-vi-qa")
qa_model = AutoModelForQuestionAnswering.from_pretrained("ancs21/xlm-roberta-large-vi-qa").to(device)
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer, device=device)

def generate_queries(context):
    input_ids = query_tokenizer.encode(context, return_tensors='pt').to(device)
    with torch.no_grad():
        queries = query_model.generate(
            input_ids=input_ids,
            max_length=150,
            num_beams=5,
            no_repeat_ngram_size=3,
            num_return_sequences=3,
        )
    return [query_tokenizer.decode(q, skip_special_tokens=True) for q in queries]

def generate_answer(question, context):
    result = qa_pipeline({"question": question, "context": context})
    return result['answer']

def clean_text(text):
    text = re.sub(r'\b(?:Trang)\s*\d+\b', '', text)
    text = re.sub(r'\b(?:Hình|Hinh)\s*.*?(?=\n|$)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

reader = PdfReader(pdf_file)
question_existed = []
counter = 1000
ocr_reader = easyocr.Reader(['vi'], gpu=True)  # Disable GPU for EasyOCR
qa_pairs = []

for i in range(118, len(reader.pages)):
    page = reader.pages[i]
    text = page.extract_text() or fromPDFtoImg(pdf_file, i)
    text = clean_text(text)
    sentences_list = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = chunk_to_sub_paragraph(text) if len(sentences_list) > 1 else sentences_list

    with open("log.txt", "a") as f:
        f.write(f"All text: {text}, Chunks: {chunks}\n")

    for chunk in chunks:
        if not chunk.strip() or chunk.isnumeric():
            continue
        questions = generate_queries(chunk)
        questions_set = set(questions)
        
        for question in questions_set:
            counter += 1
            if not question.strip() or question in question_existed:
                continue
            answer = generate_answer(question, chunk)
            if verify_qa(chunk, question, answer):
                qa_pairs.append({
                    "id": f"{file_name}_p_{i+1}_{counter}",
                    "question": question,
                    "answer": answer,
                    "context": chunk
                })
                question_existed.append(question)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=4)

    print(f"QA pairs saved to {output_file}")
