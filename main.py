import os
import pickle

def build_full_data(root_dir):
    """
    Reads validation_generations.pkl from each run directory under root_dir and
    returns a dictionary in the form:
    {
        model_name: {
            dataset_name: (list_of_questions, list_of_list_of_answers)
        }
    }
    """
    full_data = {}

    # Expecting subdirectories in format: modelname__datasetname
    for run_dir in os.listdir(root_dir):
        run_path = os.path.join(root_dir, run_dir)
        if not os.path.isdir(run_path):
            continue

        try:
            model_name, dataset_name = run_dir.split("__")
        except ValueError:
            print(f"Skipping directory with unexpected name format: {run_dir}")
            continue

        pkl_path = os.path.join(run_path, "files/validation_generations.pkl")
        if not os.path.exists(pkl_path):
            print(f"Missing file: {pkl_path}")
            continue

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            
        print(data.keys())

        questions = [item["question"] for item in data.values()]
        answers = [[ans[0] for ans in item["responses"]] for item in data.values()]
        
        # print(f"Loaded {len(questions)} questions and {len(answers)} answers from {run_dir}")

        if model_name not in full_data:
            full_data[model_name] = {}

        full_data[model_name][dataset_name] = {
            "questions": questions,
            "answers": answers
        }

    return full_data

# Example usage
if __name__ == "__main__":
    root_dir = "data"  # Replace with actual path
    full_data = build_full_data(root_dir)

    # Optional: Print summary
    for model, datasets in full_data.items():
        print(f"Model: {model}")
        for dataset, (questions, answers) in datasets.items():
            print(f"  Dataset: {dataset} | Questions: {len(questions)} | Answers per Q: {len(answers[0])}")
    
    # Save the full data to a file
    with open("full_data.pkl", "wb") as f:
        pickle.dump(full_data, f)
    print("Full data saved to full_data.pkl")
