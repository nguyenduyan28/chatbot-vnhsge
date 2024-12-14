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
def initialize_gpt4all_model(model_path):
    try:
        return GPT4All(model_name=model_path)
    except Exception as e:
        print(f"Error initializing model {model_path}: {e}")
        return None

# Function to verify a QA pair using a GPT4All model
def verify_question_with_gpt4all(model, context, question, answer):
    try:
        messages = [
            {"role": "system", "content": f"You are an expert examiner specializing in civic education knowledge assessment. Your task is to evaluate whether the following question and answer are accurate based on the provided context.\n\n"},
            {"role": "user", "content": (
                f"    *Context:* {context}\n\n"
                f"    *Question:* {question}\n\n"
                f"    *Answer:* {answer}\n\n"
                f"Carefully assess the answer's correctness considering only the provided context. Respond with *\"Correct\"* if the answer accurately aligns with the context. Respond with *\"Incorrect\"* if the answer is factually or contextually wrong. Provide only *\"Correct\"* or *\"Incorrect\"* as your answer."
            )}
        ]
        response = model.generate(messages).strip()
        return response
    except Exception as e:
        print(f"Error during verification: {e}")
        return "Error"

# Function to save verified QA pairs to a file
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
                initialized_models = {
                    model_name: initialize_gpt4all_model(model_path)
                    for model_name, model_path in models.items()
                }

                # Filter out uninitialized models
                initialized_models = {
                    name: model for name, model in initialized_models.items() if model is not None
                }

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
                        results = []

                        # Use each model to verify the QA pair
                        for model_name, model in initialized_models.items():
                            result = verify_question_with_gpt4all(model, context, question, answer)
                            if result in ["Correct", "Incorrect"]:
                                results.append(result)

                        # Verify consensus (e.g., majority agreement among models)
                        if results.count("Correct") > results.count("Incorrect"):
                            verified_qa_pairs.append({
                                "id": qa_id,
                                "question": question,
                                "answer": answer,
                                "context": context
                            })

                # Save verified QA pairs to a new file
                if verified_qa_pairs:
                    save_verified_qa_pairs(verified_qa_pairs, number_of_pages)
                    print(f"Verified QA pairs saved to verified_qa_pairs_{number_of_pages}.json")
                else:
                    print(f"No valid QA pairs found in file: {file_name}")

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {file_name}: {e}")
