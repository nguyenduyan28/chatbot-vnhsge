import os
import json
from gpt4all import GPT4All

# Define the folder containing the JSON files
folder_path = "testing"
output_folder = "verification"

# Paths to GPT4All models
models = {
    "mistral_openorca": "mistral-7b-openorca.gguf2.Q4_0.gguf",
    "gpt4all_falcon": "gpt4all-falcon-newbpe-q4_0.gguf",
    "ghost_7b_v0_9_1": "ghost-7b-v0.9.1-Q4_0.gguf"
}

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to initialize a GPT4All model
def initialize_gpt4all_model(name):
    try:
        return GPT4All(model_name=name)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None

# Function to verify a question using a GPT4All model
def verify_question_with_gpt4all(model, question, context):
    try:
        prompt = f"Verify the following question based on the context, Yes or No only:\n\nQuestion: {question}\nContext: {context}"
        response = model.generate(prompt, temp = 0.7)
        return response.strip()
    except Exception as e:
        return f"Error during verification: {e}"

# Function to check if a response indicates validity
def is_valid(response):
    # Define logic to determine if the response indicates validity
    return "yes" in response.lower()  # Adjust based on model responses

# Function to save valid QA pairs to a new file
def save_verified_qa_pairs(qa_pairs, number_of_pages):
    verified_file_path = os.path.join(output_folder, f"verified_qa_pairs_{number_of_pages}.json")
    with open(verified_file_path, "w", encoding="utf-8") as output_file:
        json.dump(qa_pairs, output_file, ensure_ascii=False, indent=4)

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

                # Initialize all models
                initialized_models = {}
                for model_name, model_path in models.items():
                    model = initialize_gpt4all_model(model_path)
                    if model:
                        initialized_models[model_name] = model

                if not initialized_models:
                    print("No models were successfully initialized.")
                    continue

                # Iterate over QA pairs and verify
                for qa_pair in data:
                    question = qa_pair.get("question", "")
                    context = qa_pair.get("context", "")
                    answer = qa_pair.get("answer", "")
                    qa_id = qa_pair.get("id", "N/A")

                    if question and context and answer:
                        valid_models = []

                        # Use each model to verify the question
                        for model_name, model in initialized_models.items():
                            result = verify_question_with_gpt4all(model, question, context)
                            if is_valid(result):
                                valid_models.append(model_name)

                        # If at least 2 out of 3 models agree, mark the QA pair as valid
                        if len(valid_models) >= 0:  # Adjust threshold as per your requirement
                            verified_qa_pairs.append({
                                "id": qa_id,
                                "question": question,
                                "answer": answer,
                                "context": context
                            })
                            print(
                                f"Verified (by models): {', '.join(valid_models)}\n"
                                f"QA ID: {qa_id}\nQuestion: {question}\n"
                                f"Context: {context}\nResult: {result}\n"
                            )


                # If there are verified QA pairs, save them to the new file
                if verified_qa_pairs:
                    save_verified_qa_pairs(verified_qa_pairs, number_of_pages)
                    print(f"Verified QA pairs saved to verified_qa_pairs_{number_of_pages}.json")
                else:
                    print(f"No valid QA pairs found in file: {file_name}")

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {file_name}: {e}")