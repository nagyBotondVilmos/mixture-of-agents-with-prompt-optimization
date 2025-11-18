import json
import random
from typing import Dict, List
from pathlib import Path
import traceback

class LeetcodeDataset:
    """Manages a dataset of leetcode problems for training"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.problems = []
        self.load_dataset()
        
    def load_dataset(self) -> bool:
        """Load the dataset of problems"""
        try:
            dataset_file = Path(self.dataset_path)
            if not dataset_file.exists():
                print(f"Dataset file not found: {self.dataset_path}")
                return False
                
            with open(dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                self.problems = data
            elif "problems" in data:
                self.problems = data["problems"]
            else:
                print("Invalid dataset format: missing 'problems' key")
                return False
                
            print(f"Loaded {len(self.problems)} problems from dataset")
            return True
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            traceback.print_exc()
            return False
            
    def get_random_problem(self) -> Dict:
        """Get a random problem from the dataset"""
        if not self.problems:
            return None
        return random.choice(self.problems)
        
    def get_random_problems(self, count: int) -> List[Dict]:
        """Get a specified number of random problems"""
        if not self.problems:
            return []
        if count >= len(self.problems):
            return self.problems.copy()
        return random.sample(self.problems, count)