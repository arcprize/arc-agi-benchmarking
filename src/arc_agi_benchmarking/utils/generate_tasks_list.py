import os
from typing import List
import argparse

def generate_task_list_from_dir(task_dir: str, output_file: str) -> List[str]:
    task_list = []
    for task_file in os.listdir(task_dir):
        task_name = os.path.splitext(task_file)[0]
        task_list.append(task_name)

    # Create the directory for the output file if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(f"{output_file}.txt", "w") as f:
        for task in task_list:
            f.write(f"{task}\n")

    return task_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str, help="Task directory")
    parser.add_argument("--output_file", type=str, help="Output file")
    args = parser.parse_args()

    generate_task_list_from_dir(args.task_dir, args.output_file)
