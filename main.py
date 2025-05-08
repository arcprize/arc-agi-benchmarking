import json
from arc_agi_testing.adapters import ProviderAdapter, AnthropicAdapter, OpenAIAdapter, DeepseekAdapter, GeminiAdapter, HuggingFaceFireworksAdapter, FireworksAdapter, GrokAdapter
from dotenv import load_dotenv
import arc_agi_testing.utils as utils
from arc_agi_testing.schemas import ARCTaskOutput, ARCPair, Attempt
from arc_agi_testing.prompts.prompt_manager import convert_task_pairs_to_prompt
from arc_agi_testing.utils.parsing import parse_and_validate_json
from typing import List, Any, Optional
import os
import argparse

load_dotenv()

class ARCTester:
    def __init__(self, config: str, save_submission_dir: str, overwrite_submission: bool, print_submission: bool, num_attempts: int, retry_attempts: int, print_logs: bool):
        self.config = config
        self.model_config = utils.read_models_config(config)
        self.provider = self.init_provider(self.model_config.provider)
        self.save_submission_dir = save_submission_dir
        self.overwrite_submission = overwrite_submission
        self.print_submission = print_submission
        self.num_attempts = num_attempts
        self.retry_attempts = retry_attempts
        self.print_logs = print_logs

    def init_provider(self, provider_name: str) -> ProviderAdapter:
        if provider_name == "anthropic":
            return AnthropicAdapter(self.config)
        elif provider_name == "openai":
            return OpenAIAdapter(self.config)
        elif provider_name == "deepseek":
            return DeepseekAdapter(self.config)
        elif provider_name == "gemini":
            return GeminiAdapter(self.config)
        elif provider_name == "huggingfacefireworks":
            return HuggingFaceFireworksAdapter(self.config)
        elif provider_name == "fireworks":
            return FireworksAdapter(self.config)
        elif provider_name == "grok":
            return GrokAdapter(self.config)
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")
        
    def print_log(self, message: str):
        if self.print_logs:
            print(message)

    def predict_task_output(self, training_pairs: List[ARCPair], test_input: ARCPair, task_id: str, test_id: str, pair_index: int):
        """
        Given a task, predict the test output. This reponse may need parsing.

        Convert the training pairs and test pairs into a prompt
        Give the prompt to the LLM
        return the response
        """

        # Convert the training pairs and test pairs into a prompt
        prompt = convert_task_pairs_to_prompt(training_pairs, test_input)

        self.print_log(f"Making prediction for task {task_id}, test {test_id}")
        response: Attempt = self.provider.make_prediction(prompt, task_id=task_id, test_id=test_id, pair_index=pair_index)

        return response

    def get_task_prediction(self, training_pairs: List[ARCPair], test_input: ARCPair, task_id: str, test_id: str, pair_index: int) -> Attempt:
        """
        Modified to return the full Attempt object instead of just the parsed answer
        Uses the refactored parsing logic from arc_agi_testing.parsing
        """
        # Get the initial response as an Attempt object
        attempt: Attempt = self.predict_task_output(training_pairs, test_input, task_id, test_id, pair_index)

        try:
            # Always use the last choice in the array (which should be the assistant's response)
            if attempt.metadata.choices:
                last_choice = attempt.metadata.choices[-1]
                if last_choice.message.content:
                    parsed_answer = parse_and_validate_json(
                        response=last_choice.message.content, 
                        provider_extractor=self.provider.extract_json_from_response
                    )
                else:
                    raise ValueError("Assistant response is empty")
            else:
                raise ValueError("No choices found in response")
            
            # Update the answer in the original attempt - now accepts List[List[int]]
            attempt.answer = parsed_answer
            return attempt
        except (json.JSONDecodeError, ValueError) as e: # Catch parsing and validation errors
            self.print_log(f"Parsing/Validation failed: {e}")
            raise

    def generate_task_solution(self, data_dir, task_id):
        """
        data_dir: str, the directory of the data set to run
        task_id: str, the specific task to run. If None, run all tasks.
        num_attempts: int the number of times to attempt a prediction. The official competition has 2 attempts.
        retry_attempts: int the number of times to retry a prediction if it fails
        save_submission: bool, whether to save the submission to a file after each task
        """
        
        self.print_log(f"Running task {task_id}")
        utils.validate_data(data_dir, task_id)

        # Use the config name as the test_id
        test_id = self.config
        
        self.print_log(f"Using test_id: {test_id}")

        # Logic for overwrite. If save_submission_dir is provided, check if the submission already exists
        if self.save_submission_dir and utils.submission_exists(self.save_submission_dir, task_id) and not self.overwrite_submission:
            self.print_log(f"Submission for task {task_id} already exists, skipping")
            return
        
        task_attempts = []

        train_pairs = utils.get_train_pairs_from_task(data_dir, task_id)
        test_input = utils.get_test_input_from_task(data_dir, task_id)

        # Go through each test pair to get a prediction. 96% of challenges have 1 pair.
        for t, pair in enumerate(test_input):
            current_test_id = str(t)
            self.print_log(f"Starting task {task_id}, Pair #{t+1}")
            pair_attempts = {}

            # Run through each prediction attempt
            for attempt in range(1, self.num_attempts + 1):
                attempt_key = f"attempt_{attempt}"
                pair_attempts[attempt_key] = None

                for retry in range(self.retry_attempts):
                    try:
                        self.print_log(f"    Predicting attempt #{attempt}, retry #{retry + 1}")
                        # Now storing the full attempt object with task_id and test_id
                        attempt_obj = self.get_task_prediction(
                            training_pairs=train_pairs,
                            test_input=pair,
                            task_id=task_id,
                            test_id=test_id,
                            pair_index=t
                        )

                        if attempt_obj is not None:
                            self.print_log(f"    Prediction: {attempt_obj.answer}")
                            pair_attempts[attempt_key] = attempt_obj.model_dump(mode='json')
                            break
                    except Exception as e:
                        self.print_log(f"Retrying: {e}")

                    if retry == self.retry_attempts - 1:
                        self.print_log(f"    All retries failed for attempt #{attempt}")

            # Only append non-None attempts
            if any(v is not None for v in pair_attempts.values()):
                task_attempts.append(pair_attempts)

        if task_attempts:
            if self.print_submission:
                self.print_log(f"Submission for task {task_id}:\n{task_attempts}")

            if self.save_submission_dir:
                utils.save_submission(self.save_submission_dir, task_id, task_attempts)
        else:
            self.print_log(f"No valid predictions for task {task_id}, skipping submission")

        return task_attempts if task_attempts else None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARC Tester")
    parser.add_argument("--data_dir", type=str, help="Data set to run. Configure in config/config.json")
    parser.add_argument("--task_id", type=str, help="Specific task ID to run")
    parser.add_argument("--config", type=str, required=True, help="Configuration name (e.g., 'o1_high', 'gemini_short_response')")
    parser.add_argument(
        "--save_submission_dir",
        type=str,
        metavar="FOLDER_NAME",
        help="Folder name to save the submissions under Ex: 'submissions/o1_high'"
    )
    parser.add_argument("--overwrite_submission", action="store_true", help="Overwrite the submission if it already exists")
    parser.add_argument("--print_submission", action="store_true", help="Print the submission to the console after each task")
    parser.add_argument("--task_set", type=str, default="public_eval", choices=["public_eval", "public_training"], help="Task set to run")
    parser.add_argument("--num_attempts", type=int, default=2, help="Number of attempts for each prediction")
    parser.add_argument("--retry_attempts", type=int, default=2, help="Number of retry attempts for failed predictions")
    parser.add_argument("--print_logs", action="store_true", help="Disable printing logs to console (default: False)")
    args = parser.parse_args()

    arc_solver = ARCTester(
        config=args.config,
        save_submission_dir=args.save_submission_dir, 
        overwrite_submission=args.overwrite_submission,
        print_submission=args.print_submission,
        num_attempts=args.num_attempts,
        retry_attempts=args.retry_attempts,
        print_logs=args.print_logs
    )
   
    arc_solver.generate_task_solution(
        data_dir=args.data_dir,
        task_id=args.task_id
    )
