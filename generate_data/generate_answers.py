"""
Script to generate answers from a dataset using Hugging Face models.
This script loads a dataset, creates prompts from each sample, calls the model's predict
method, and then stores the results in a dictionary.
"""

import os
import random
import logging
import torch

from load_dataset import load_ds
from models import HuggingfaceModel

# Configure logging
logging.basicConfig(level=logging.INFO)

def split_dataset(dataset):
    """Get indices of answerable and unanswerable questions."""

    def clen(ex):
        return len(ex["answers"]["text"])

    answerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) > 0]
    unanswerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) == 0]

    # union == full dataset
    assert set(answerable_indices) | set(
        unanswerable_indices) == set(range(len(dataset)))
    # no overlap
    assert set(answerable_indices) - \
        set(unanswerable_indices) == set(answerable_indices)

    return answerable_indices, unanswerable_indices

def generate_answers(model, train_dataset, validation_dataset, num_samples=10):
    results = {}
    total = len(dataset)
    indices = random.sample(range(total), min(num_samples, total))
    logging.info("Using %d examples out of %d", len(indices), total)

        
    for idx in indices:
        example = dataset[idx]
        qid = example["id"]
        question = example["question"]
        context = example["context"]
        prompt = f"Context : {context}\n Question : {question}\nAnswer:"
        logging.info("Processing example id %s", qid)
        logging.info("Processing question: %s", question)

        # Use a low temperature for the most likely answer.
        predicted_answer, token_log_likelihoods, embedding = model.predict(prompt, temperature=0.1)
        
        # Store the results
        results[qid] = {
            "question": question,
            "context": context,
            "most_likely_answer": {
                "response": predicted_answer,
                "token_log_likelihoods": token_log_likelihoods,
                "embedding": embedding.cpu() if isinstance(embedding, torch.Tensor) else embedding
            },
            "reference": example.get("answers", {}).get("text", "")
        }
    return results

def main():
    # Dataset Name ("svamp", "squad", "trivia_qa", "bioasq", "nq")
    dataset_name = "svamp"
    num_samples = 5  

    train_dataset, validation_dataset = load_ds(dataset_name, add_options=False, seed=42)
   
    if not isinstance(train_dataset, list):
        logging.error("Dataset format not recognized.")
        return


    model_name = "Llama-3.2-1B"  
    max_new_tokens = 100
    model = HuggingfaceModel(model_name, stop_sequences="default", max_new_tokens=max_new_tokens)

    # Generate answers.
    generations = generate_answers(model, train_dataset, num_samples=num_samples)
    
    # # For demonstration, print the results.
    # for qid, data in generations.items():
    #     print("="*40)
    #     print(f"ID: {qid}")
    #     print("Question:", data["question"])
    #     print("Context:", data["context"])
    #     print("Answer:", data["most_likely_answer"]["response"])
    #     print("Log Likelihoods:", data["most_likely_answer"]["token_log_likelihoods"])
    #     print()

if __name__ == "__main__":
    main()