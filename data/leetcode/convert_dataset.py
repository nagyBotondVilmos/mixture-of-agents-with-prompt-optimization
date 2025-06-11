import csv
import json
import argparse
import os
from pathlib import Path
import sys

def convert_csv_to_json(csv_file, json_file, include_solutions=True):
    """
    Convert CSV file with Leetcode problems to JSON format expected by trainer.py
    
    CSV format expected:
    id,problem_description,sol
    
    JSON format output:
    [
        {
            "id": "123",
            "description": "Problem description text",
            "solution": "Python solution code"  # Optional, included if include_solutions is True
        },
        ...
    ]
    """
    try:
        problems = []
        
        # Read CSV file
        with open(csv_file, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            
            # Check if required columns exist
            header = csv_reader.fieldnames
            if not header:
                print("Error: CSV file appears to be empty")
                return False
                
            required_fields = ['id', 'problem_description']
            if include_solutions:
                required_fields.append('sol')
                
            missing_fields = [field for field in required_fields if field not in header]
            if missing_fields:
                print(f"Error: CSV is missing required fields: {', '.join(missing_fields)}")
                return False
            
            # Process each row
            for row in csv_reader:
                problem = {
                    "id": row.get('id', ''),
                    "description": row.get('problem_description', '')
                }
                
                # Add solution if requested
                if include_solutions and 'sol' in row:
                    problem["solution"] = row['sol']
                    
                # Skip if description is empty
                if not problem["description"]:
                    continue
                    
                problems.append(problem)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(json_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write JSON file
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(problems, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully converted {len(problems)} problems to {json_file}")
        return True
        
    except Exception as e:
        print(f"Error converting CSV to JSON: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert CSV dataset to JSON format for the trainer")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    parser.add_argument("--solutions", "-s", action="store_true", help="Include solutions in the output")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
        
    # Convert CSV to JSON
    success = convert_csv_to_json(args.input, args.output, args.solutions)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 