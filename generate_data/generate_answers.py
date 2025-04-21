"""
Script to generate answers from a dataset using Hugging Face models.
This script loads a dataset, creates prompts from each sample, calls the model's predict
method, and then stores the results in a dictionary.
"""

import os
import random
import logging
import torch
import pickle

from load_dataset import load_ds
from models import HuggingfaceModel

# Configure logging
logging.basicConfig(level=logging.INFO)

import re

def clean_context(context):
    return re.sub(r'\[[A-Z]+\]', '', context).strip()


def save(object, file):
    """Save an object to a file using pickle."""
    with open(file, 'wb') as f:
        pickle.dump(object, f)
    logging.info(f"Saved object to {file}")


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

def get_make_prompt():
    def make_prompt(context, question, answer, brief, brief_always):
        prompt = ''
        if brief_always:
            prompt += brief
        if (context is not None):
            prompt += f"Context: {context}\n"
        prompt += f"Question: {question}\n"
        if answer:
            prompt += f"Answer: {answer}\n\n"
        else:
            prompt += 'Answer:'
        return prompt
    return make_prompt

BRIEF_PROMPTS = {
    'default': "Answer the following question as briefly as possible.\n",
    'chat': 'Answer the following question in a single brief but complete sentence.\n'}

def get_reference(example):
    if 'answers' not in example:
        example = example['reference']
    answers = example['answers']
    answer_starts = answers.get('answer_start', [])
    reference = {'answers': {'answer_start': answer_starts, 'text': answers['text']}, 'id': example['id']}
    return reference

def construct_fewshot_prompt_from_indices(dataset, example_indices, brief, brief_always, make_prompt):
    """Given a dataset and indices, construct a fewshot prompt."""
    if not brief_always:
        prompt = brief
    else:
        prompt = ''

    for example_index in example_indices:

        example = dataset[example_index]
        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0]

        prompt = prompt + make_prompt(context, question, answer, brief, brief_always)

    return prompt

def generate_answers(model, train_dataset, validation_dataset, num_samples=10, num_generations=3, temperature=0.4):
    answerable_indices, _ = split_dataset(train_dataset)

    prompt_indices = random.sample(answerable_indices, num_samples)
    make_prompt = get_make_prompt()
    BRIEF = BRIEF_PROMPTS["default"]
    prompt = construct_fewshot_prompt_from_indices(train_dataset, prompt_indices, BRIEF, False, make_prompt)
    
    for dataset_split in ['train', 'validation']:
        if dataset_split == 'train':
            dataset = train_dataset
            possible_indices = list(set(answerable_indices))
        else:
            dataset = validation_dataset
            possible_indices = range(len(dataset))

        indices = random.sample(possible_indices, min(num_samples, len(dataset)))
        generations = {}

        for index in indices:
            example = dataset[index]
            question, context = example["question"], example["context"]
            context = clean_context(context)
            correct_answer = example["answers"]["text"]

            # Create the current input and combine with the few-shot prompt.
            current_input = make_prompt(context, question, None, BRIEF, True)
            local_prompt = current_input

            # print(local_prompt)

            full_responses = []
            num_generations = num_generations + 1  # First generation is low temperature.
            for i in range(num_generations):
                temperature = 0.1 if i == 0 else temperature
                predicted_answer, token_log_likelihoods, embedding = model.predict(local_prompt, temperature)
                
                # For the first generation, compute accuracy and store as the main answer.
                if i == 0:
                    generations[example["id"]] = {
                        "question": question,
                        "context": context,
                        "most_likely_answer": {
                            "response": predicted_answer,
                            "token_log_likelihoods": token_log_likelihoods,
                            "embedding": embedding,
                        },
                        "reference": get_reference(example)
                    }
                    print(f"Example ID: {example['id']}")
                    print(f"Question: {question}")
                    print(f"Context: {context}")
                    print(f"Predicted Answer: {predicted_answer}")
                    print(f"Correct Answer: {correct_answer}")
                else:
                    full_responses.append((predicted_answer, token_log_likelihoods, embedding, 0.0))
            generations[example["id"]]["responses"] = full_responses
        save(generations, f"{dataset_split}_generations.pkl")

def main():
    # Dataset Name ("svamp", "squad", "trivia_qa", "bioasq", "nq")
    random.seed(47)
    dataset_name = "trivia_qa"
    num_samples = 100  

    train_dataset, validation_dataset = load_ds(dataset_name, add_options=False, seed=42)
   
    if not isinstance(train_dataset, list):
        logging.error(f"Dataset format not recognized, got: {type(train_dataset)} ")
        return


    model_name = "Llama-3.2-1B"  
    max_new_tokens = 100
    model = HuggingfaceModel(model_name, stop_sequences="default", max_new_tokens=max_new_tokens)

    # Generate answers.
    generate_answers(model, train_dataset, validation_dataset, num_samples=num_samples)

if __name__ == "__main__":
    main()