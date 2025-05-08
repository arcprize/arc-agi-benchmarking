import argparse
from typing import List, Tuple
from pathlib import Path
from typing import List, Tuple, Dict
import json
from arc_agi_testing.schemas import ARCTask, TestedTask

class ARCScorer:
    def __init__(self, task_dir: str, submission_dir: str, print_logs: bool = False, results_dir: str = None):
        self.task_dir = Path(task_dir)
        self.submission_dir = Path(submission_dir)
        self.print_logs = print_logs
        self.solutions = self._load_solutions()
        self.results_dir = Path(results_dir) if results_dir else None
        self.total_cost = 0.0
        self.total_attempts = 0

    def print_log(self, message: str) -> None:
        if self.print_logs:
            print(message)

    def _load_solutions(self) -> Dict[str, List[List[List[int]]]]:
        solutions = {}
        for solution_file in self.task_dir.glob('*.json'):
            task_id = solution_file.stem
            with solution_file.open() as f:
                solutions[task_id] = json.load(f)
        return solutions

    @staticmethod
    def score_task(task: ARCTask, testing_results: TestedTask) -> Tuple[float, float, int]:
        """
        Score a task against the solutions.
        """
        
        task_score = 0
        num_pairs = len(task.test)
        task_cost = 0.0
        num_attempts = 0

        for enum_pair_index, pair_attempts in enumerate(testing_results):
            # Try to extract pair_index from the attempt key
            pair_index = None
            pair_index_found = False
            
            # First, check if pair_index is directly in the attempt data
            for attempt_key in pair_attempts:
                attempt_data = pair_attempts[attempt_key] # Get data first
                
                # Skip None attempts silently when just looking for pair_index
                if attempt_data is None: 
                    continue
                
                # Validate structure if data is present
                if not isinstance(attempt_data, dict):
                    raise TypeError(f"Task {task_id}, Attempt {attempt_key}: Expected dictionary for attempt data, got {type(attempt_data)}")
                if 'metadata' not in attempt_data:
                    raise KeyError(f"Task {task_id}, Attempt {attempt_key}: Missing 'metadata' key.")
                if 'pair_index' not in attempt_data['metadata']:
                     # Don't raise error here yet, another attempt might have it
                     # But log a warning if desired
                     self.print_log(f"    Warning: Task {task_id}, Attempt {attempt_key}: Missing 'pair_index' in 'metadata'.")
                     continue # Check next attempt for pair_index

                # If checks pass, try to use this pair_index
                pair_index = attempt_data['metadata']['pair_index']
                if 0 <= pair_index < num_pairs:  # Validate the index
                    pair_index_found = True
                    break # Found a valid index, stop searching attempts for this pair
                else:
                    # Invalid index found in this attempt, reset and check next attempt
                    pair_index = None 

            # If not found in any attempt's metadata, fall back to enumeration index
            if not pair_index_found:
                pair_index = enum_pair_index
                
            # Skip if the final pair_index (either found or fallback) is out of bounds
            if not (0 <= pair_index < num_pairs):
                self.print_log(f"    Warning: Invalid or out-of-bounds pair_index {pair_index} derived for pair {enum_pair_index} in task {task_id}, skipping pair.")
                continue
            
            # Count all attempts in this pair, regardless of whether we process them all
            num_attempts += len(pair_attempts)
            
            # Calculate costs for all attempts upfront
            for attempt_key in pair_attempts:
                attempt_data = pair_attempts[attempt_key]
                if attempt_data is None:
                    continue
                
                # Validate structure for cost calculation
                if not isinstance(attempt_data, dict):
                     # This case should have been caught by the pair_index loop, but double-check
                    raise TypeError(f"Task {task_id}, Pair {pair_index}, Attempt {attempt_key}: Expected dictionary for attempt data, got {type(attempt_data)}")
                if 'metadata' in attempt_data and 'cost' in attempt_data['metadata']:
                    attempt_cost = attempt_data['metadata']['cost'].get('total_cost', 0.0)
                    task_cost += attempt_cost
                # Optionally log or raise if metadata or cost is missing?
                else:
                    self.print_log(f"    Warning: Task {task_id}, Pair {pair_index}, Attempt {attempt_key}: Missing metadata or cost info.")
            
            pair_correct = False
            
            for attempt_key in pair_attempts:
                attempt_data = pair_attempts[attempt_key]

                if attempt_data is None:
                    self.print_log(f"    No prediction for {task_id}, pair {pair_index}, attempt {attempt_key}")
                    continue
                
                # Validate structure before checking answer
                if not isinstance(attempt_data, dict):
                     # This case should have been caught by the pair_index loop, but double-check
                    raise TypeError(f"Task {task_id}, Pair {pair_index}, Attempt {attempt_key}: Expected dictionary for attempt data, got {type(attempt_data)}")
                if 'answer' not in attempt_data:
                    raise KeyError(f"Task {task_id}, Pair {pair_index}, Attempt {attempt_key}: Missing 'answer' key.")

                # Handle empty list answer (treat as incorrect attempt)
                if attempt_data['answer'] == []:
                    self.print_log(f"    Empty list prediction for {task_id}, pair {pair_index}, attempt {attempt_key}")
                    continue

                # Check for exact match
                if attempt_data['answer'] == self.solutions[task_id]['test'][pair_index]['output']:
                    pair_correct = True
                    break

            if pair_correct:
                task_score += 1

        scoring_result = {
            "score": task_score / num_pairs if num_pairs > 0 else 0.0,
            "cost": task_cost,
            "attempts": num_attempts
        }

        return scoring_result


    def score_task_from_file(self, task_id: str, submission_path: Path) -> Tuple[float, float, int]:
        """
        Scores a single task submission against the solutions.
        Returns a dictionary containing task_score, task_cost, num_attempts.
        """
        with submission_path.open() as f:
            task_submission = TestedTask(**json.load(f))
        task = ARCTask.from_dict(self.solutions[task_id]) 
        return self.score_task(task, task_submission)


    def score_submission(self) -> Tuple[float, int]:
        """
        Read a submission from file, score it, then return the score
        """
        self.print_log(f"Scoring {self.submission_dir}\n")

        total_score = 0
        total_tasks = 0
        task_results = {}

        for submission_file in self.submission_dir.glob('*.json'):
            if submission_file.name == 'results.json':
                continue

            task_id = submission_file.stem
            scoring_result = self.score_task_from_file(task_id, submission_file)
            
            total_score += scoring_result["score"]
            total_tasks += 1
            self.total_cost += scoring_result["cost"]
            self.total_attempts += scoring_result["attempts"]
            
            task_results[task_id] = {
                "score": scoring_result['score'],
                "cost": scoring_result['cost'],
                "attempts": scoring_result['attempts']
            }

            self.print_log(f"    Task {task_id} score: {scoring_result['score']:.2f}, cost: ${scoring_result['cost']:.4f}, attempts: {scoring_result['attempts']}")

        # Calculate average costs
        avg_cost_per_task = self.total_cost / total_tasks if total_tasks > 0 else 0
        avg_cost_per_attempt = self.total_cost / self.total_attempts if self.total_attempts > 0 else 0

        # Only save results if results_dir is provided
        if self.results_dir:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            results_file = self.results_dir / "results.json"
            self.print_log(f"Saving results to {results_file}")
            with results_file.open("w") as f:
                json.dump({
                    "score": total_score,
                    "total_tasks": total_tasks,
                    "total_cost": self.total_cost,
                    "total_attempts": self.total_attempts,
                    "avg_cost_per_task": avg_cost_per_task,
                    "avg_cost_per_attempt": avg_cost_per_attempt,
                    "task_results": task_results
                }, f, indent=4)

        # Calculate and print percentage score
        percentage_score = (total_score / total_tasks * 100) if total_tasks > 0 else 0
        print(f"\nFinal Score: {percentage_score:.2f}% ({total_score:.2f}/{total_tasks})")
        print(f"Total Cost: ${self.total_cost:.4f}")
        print(f"Average Cost per Task: ${avg_cost_per_task:.4f}")
        print(f"Average Cost per Attempt: ${avg_cost_per_attempt:.4f}")
        print(f"Total Attempts: {self.total_attempts}")

        return total_score, total_tasks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARC Tester")

    parser.add_argument("--task_dir", type=str, help="Task directory which contains the full tasks (including solutions)")
    parser.add_argument("--submission_dir", type=str, help="Submission directory which contains the submissions to score")
    parser.add_argument("--print_logs", action="store_true", help="Printing logs to console (default: False)")
    parser.add_argument("--results_dir", type=str, help="Results directory to save the results (default: None)")

    args = parser.parse_args()

    arc_scorer = ARCScorer(
        task_dir=args.task_dir,
        submission_dir=args.submission_dir,
        print_logs=args.print_logs,
        results_dir=args.results_dir
    )

    arc_scorer.score_submission()