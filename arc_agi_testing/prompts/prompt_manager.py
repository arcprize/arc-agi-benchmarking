import json
from arc_agi_testing.schemas import ARCPair
from typing import List

def _load_prompt(prompt_name: str) -> str:
    """
    Load a prompt from the prompts directory
    """

    return open(f"arc_agi_testing/prompts/{prompt_name}.txt", "r").read()

def convert_task_pairs_to_prompt(training_pairs: List[ARCPair], test_input: ARCPair) -> str:
    """
    Convert the training pairs to a prompt
    """

    prompt_template = _load_prompt("system_prompt")

    training_examples = ""
    for i, pair in enumerate(training_pairs):
        training_examples += f"--Example {i}-- \n\n INPUT: \n\n"
        training_examples += json.dumps(pair.input) + "\n\n"
        training_examples += f"OUTPUT: \n\n"
        training_examples += json.dumps(pair.output) + "\n\n"

    test_input_str = json.dumps(test_input.input)

    return prompt_template.format(training_examples=training_examples, test_input=test_input_str)