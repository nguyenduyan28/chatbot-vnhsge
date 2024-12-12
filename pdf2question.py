from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, pipeline, XLMRobertaForQuestionAnswering
import re
import torch
import spacy
import torch
import os
import json
import easyocr
from pdf2image import convert_from_path
from extract_ocr import fromPDFtoImg
device = torch.device('mps')
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
model = HuggingFaceEmbeddings(model_name = 'dangvantuan/vietnamese-embedding')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

nlp = spacy.load("vi_core_news_lg")


def is_valid_answer(answer, context):
    if len(answer) < 3 or answer.lower() in context.lower():
        return False
    return True


from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer('dangvantuan/vietnamese-embedding')


def smt2(text):
    chunker = SemanticChunker(embeddings=model)
    chunks = chunker.split_text(text)
    return chunks


def recursive_chunking(text, chunk_size=300, overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return text_splitter.split_text(text)
query_model_name = "doc2query/msmarco-vietnamese-mt5-base-v1"
query_tokenizer = AutoTokenizer.from_pretrained(query_model_name)
query_model = AutoModelForSeq2SeqLM.from_pretrained(query_model_name)



qa_tokenizer = AutoTokenizer.from_pretrained("ancs21/xlm-roberta-large-vi-qa")
qa_model = AutoModelForQuestionAnswering.from_pretrained("ancs21/xlm-roberta-large-vi-qa")

qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer, device=1)

def generate_queries(context):
    input_ids = query_tokenizer.encode(context, return_tensors='pt')
    with torch.no_grad():
        queries = query_model.generate(
            input_ids=input_ids,
            max_length=150,  # Adjust length
            num_beams=5,  # Increase beam size
            no_repeat_ngram_size=3,  # Avoid repetitions
            num_return_sequences=5,  # More queries for variety
        )
    return [query_tokenizer.decode(q, skip_special_tokens=True) for q in queries]

def generate_answer(question, context):
    result = qa_pipeline({"question": question, "context": context})
    return result['answer']




pdf_file = "ocr_material/Lich su 12.pdf"
reader = PdfReader(pdf_file)
question_existed = []
counter = 1
ocr_reader = easyocr.Reader(['vi'], gpu=True)
for i in range(len(reader.pages)):
    page = reader.pages[i]
    text = page.extract_text()
    if (len(text) == 0):
        text = fromPDFtoImg(pdf_file, i)
    chunks = recursive_chunking(text)
    qa_pairs = []

    for chunk in chunks:
        if not chunk.strip() or chunk.isnumeric():
            continue

        if (chunk is None):
            continue
        questions = generate_queries(chunk)
        print(questions)
        questions_set = set(questions)
        
        for question in questions_set:
            counter += 1
            if not question.strip():
                continue  # Skip empty questions
            if (question not in question_existed):
                answer = generate_answer(question, chunk)
                qa_pairs.append({"id": f"p_{i + 1}_{counter}","question": question, "answer": answer, "context": chunk})
                question_existed.append(question)


    # Save the results to a JSON file
    output_file = f"qa_pairs_ocr/qa_pairs_{i + 1}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=4)

    print(f"QA pairs saved to {output_file}")