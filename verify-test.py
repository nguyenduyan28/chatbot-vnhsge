import os
import json
from gpt4all import GPT4All

# Define the folder containing the JSON files
folder_path = "testing"
output_folder = "verification"

# Path to the GPT4All ghost model
ghost_model_path = "ghost-7b-v0.9.1-Q4_0.gguf"

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to initialize the GPT4All ghost model
def initialize_gpt4all_ghost_model(model_path):
    try:
        return GPT4All(model_name=model_path)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None

# Function to verify a question using the GPT4All ghost model
def verify_question_with_ghost(model, question, context):
    try:
        prompt = f"Yes or No, Use English to answer, Is this question valid based on the context:\n\nQuestion: {question}\nContext: {context}"
        response = model.generate(prompt, max_tokens=200)
        return response.strip()
    except Exception as e:
        return f"Error during verification: {e}"

# Function to check if a response indicates validity
def is_valid(response):
    return "yes" in response.lower()  # Adjust based on model responses

# Function to save valid QA pairs to a new file
def save_verified_qa_pairs(qa_pairs, number_of_pages):
    verified_file_path = os.path.join(output_folder, f"verified_qa_pairs_{number_of_pages}.json")
    with open(verified_file_path, "w", encoding="utf-8") as output_file:
        json.dump(qa_pairs, output_file, ensure_ascii=False, indent=4)

# Initialize the ghost model
ghost_model = initialize_gpt4all_ghost_model(ghost_model_path)
if not ghost_model:
    print("Failed to initialize the ghost model. Exiting.")
    exit()

# Main process
for file_name in os.listdir(folder_path):
    if file_name.startswith("qa_pairs_") and file_name.endswith(".json"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
                verified_qa_pairs = []

                # Extract the number of pages from the file name
                number_of_pages = file_name.replace("qa_pairs_", "").replace(".json", "").strip()

                # Iterate over QA pairs and verify
                for qa_pair in data:
                    question = qa_pair.get("question", "")
                    context = qa_pair.get("context", "")
                    answer = qa_pair.get("answer", "")
                    qa_id = qa_pair.get("id", "N/A")

                    if question and context and answer:
                        result = verify_question_with_ghost(ghost_model, question, context)
                        print(f"QA ID: {qa_id}, Response: {result}")

                        # Check validity based on the ghost model's response
                        if is_valid(result):
                            verified_qa_pairs.append({
                                "id": qa_id,
                                "question": question,
                                "answer": answer,
                                "context": context
                            })

                # If there are verified QA pairs, save them to the new file
                if verified_qa_pairs:
                    save_verified_qa_pairs(verified_qa_pairs, number_of_pages)
                    print(f"Verified QA pairs saved to verified_qa_pairs_{number_of_pages}.json")
                else:
                    print(f"No valid QA pairs found in file: {file_name}")

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {file_name}: {e}")
