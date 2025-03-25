import json
import re

# Input file (example.json)
input_file = "./logs/humaneval_0_163_final.json"

# Output file (sample_file.jsonl)
output_file = "humaneval_0_163_test.jsonl"

def extract_samples(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    task_id = 0  # Start task_id from 0

    for item in data:
        response = item.get("Response", "")
        # Extract content inside boxed{*}
        match = re.search(r'boxed\{(.*?)\}', response, re.DOTALL)
        if match:
            completion = match.group(1).strip()  # Extracted content inside boxed{}
        else:
            completion = ""  # Default to empty if no match found
        
        sample = {
            "task_id": f"HumanEval/{task_id}",  # Assign task_id as "HumanEval/x"
            "completion": completion  # Use extracted content as completion
        }
        samples.append(sample)
        task_id += 1

    # Write to output file in JSONL format
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Extracted {len(samples)} samples to {output_file}")

# Call the function
extract_samples(input_file, output_file)