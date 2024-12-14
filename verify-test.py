import os
import json
from gpt4all import GPT4All

# Define the folder containing the JSON files
folder_path = "qa_pairs_ocr"
output_folder = "verification"

# Path to the GPT4All ghost model
ghost_model_path = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"

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

# Function to verify a batch of questions using the GPT4All ghost model
def verify_batch_with_ghost(model, batch):
    # Construct the prompt for the batch
    prompt = "\n".join([
        (
            "Evaluate the following based on the provided context. "
            "Reply with 'Correct' or 'Incorrect' only.\n"
            f"Context: {qa['context']}\n"
            f"Question: {qa['question']}\n"
            f"Answer: {qa['answer']}\n"
        ) for qa in batch
    ])

    # Generate response from the model
    response = model.generate(prompt, max_tokens=1024)

    # Split the response into individual lines
    responses = response.strip().split("\n")
    return responses

# Function to check if a response indicates validity
def is_valid(response):
    return "correct" in response.lower()

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

batch_size = 5  # Define the size of each batch
for file_name in os.listdir(folder_path):
    if file_name.startswith("qa_pairs_") and file_name.endswith(".json"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
                verified_qa_pairs = []

                # Extract the number of pages from the file name
                number_of_pages = file_name.replace("qa_pairs_", "").replace(".json", "").strip()

                # Process data in batches
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]

                    # Verify the batch
                    responses = verify_batch_with_ghost(ghost_model, batch)

                    # Match responses with QA pairs
                    for qa, response in zip(batch, responses):
                        qa_id = qa.get("id", "N/A")

                        if is_valid(response):
                            qa["model_response"] = response  # Save the model's response
                            verified_qa_pairs.append(qa)
                            print(f"QA ID: {qa_id}, Response: yes")
                        else:
                            print(f"QA ID: {qa_id}, Response: no")

                # If there are verified QA pairs, save them to the new file
                if verified_qa_pairs:
                    save_verified_qa_pairs(verified_qa_pairs, number_of_pages)
                    print(f"Verified QA pairs saved to verified_qa_pairs_{number_of_pages}.json")
                else:
                    print(f"No valid QA pairs found in file: {file_name}")

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {file_name}: {e}")
