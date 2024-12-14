import os
import json
from gpt4all import GPT4All

ghost_model_path = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"

def initialize_gpt4all_ghost_model(model_path):
    try:
        return GPT4All(model_name=model_path)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None
def verify_single_with_ghost(model, context, question, answer):
    prompt = (
        "Evaluate the following based on the provided context. "
        "Reply with 'Correct' or 'Incorrect' only.\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
    )
    # Generate response from the model
    response = model.generate(prompt, max_tokens=1024).strip().lower()
    return "correct" in response

ghost_model = initialize_gpt4all_ghost_model(ghost_model_path)
if not ghost_model:
    print("Failed to initialize the ghost model. Exiting.")
    exit()

def verify_qa(context, question, answer):
  if verify_single_with_ghost(ghost_model, context, question, answer):
    return True
  else:
    return False
