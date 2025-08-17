import os
import json
import torch
import numpy as np
from typing import List, Dict, Any
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llms.generate import Generator
from llms.config import TrainArgs

class HumanEvalEvaluator:
    """Comprehensive HumanEval benchmark evaluator for code generation models."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3-8B", use_local_model: bool = False):
        """
        Initialize the evaluator.
        
        Args:
            model_name: HuggingFace model name or local path
            use_local_model: Whether to use local model implementation
        """
        self.model_name = model_name
        self.use_local_model = use_local_model
        
        if use_local_model:
            self.setup_local_model()
        else:
            self.setup_hf_model()
            
        self.dataset = load_dataset("openai_humaneval")
        
    def setup_local_model(self):
        """Setup local model implementation."""
        args = TrainArgs()
        self.generator = Generator(args)
        self.tokenizer = None  # Uses SentencePiece from generator
        
    def setup_hf_model(self):
        """Setup HuggingFace model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.generator = None
        
    def generate_completion(self, prompt: str, max_length: int = 512) -> str:
        """Generate code completion for a given prompt."""
        if self.use_local_model:
            return self._generate_local(prompt, max_length)
        else:
            return self._generate_hf(prompt, max_length)
    
    def _generate_local(self, prompt: str, max_length: int) -> str:
        """Generate using local model implementation."""
        completions = self.generator.text_completion(
            [prompt], 
            temperature=0.2, 
            top_p=0.95
        )
        return completions[0] if completions else ""
    
    def _generate_hf(self, prompt: str, max_length: int) -> str:
        """Generate using HuggingFace model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        completion = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return completion
    
    def extract_function_body(self, completion: str) -> str:
        """Extract the function body from the completion."""
        lines = completion.split('\n')
        function_lines = []
        in_function = False
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
            # Check if we're starting a function definition
            if stripped.startswith('def ') and not in_function:
                in_function = True
                indent_level = len(line) - len(line.lstrip())
                continue
                
            if in_function:
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level and stripped:
                    break
                function_lines.append(line)
                
        return '\n'.join(function_lines)
    
    def evaluate_single_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single HumanEval problem."""
        prompt = problem['prompt']
        test_cases = problem['test']
        
        # Generate completion
        completion = self.generate_completion(prompt)
        function_body = self.extract_function_body(completion)
        
        # Combine prompt and completion
        full_code = prompt + function_body
        
        # Execute test cases
        results = []
        for test_case in test_cases:
            try:
                # Create a safe execution environment
                local_vars = {}
                exec(full_code, {}, local_vars)
                
                # Extract function name from prompt
                function_name = prompt.split('def ')[1].split('(')[0].strip()
                func = local_vars.get(function_name)
                
                if func is None:
                    results.append(False)
                    continue
                
                # Execute test case
                exec(test_case, {}, local_vars)
                results.append(True)
                
            except Exception as e:
                results.append(False)
        
        return {
            'task_id': problem['task_id'],
            'completion': completion,
            'function_body': function_body,
            'full_code': full_code,
            'test_results': results,
            'passed': sum(results),
            'total_tests': len(results),
            'pass_rate': sum(results) / len(results) if results else 0.0
        }
    
    def evaluate_all(self, max_problems: int = None) -> Dict[str, Any]:
        """Evaluate all HumanEval problems."""
        problems = self.dataset['test']
        if max_problems:
            problems = problems.select(range(min(max_problems, len(problems))))
        
        results = []
        for i, problem in enumerate(problems):
            print(f"Evaluating problem {i+1}/{len(problems)}: {problem['task_id']}")
            result = self.evaluate_single_problem(problem)
            results.append(result)
        
        # Calculate overall metrics
        total_passed = sum(r['passed'] for r in results)
        total_tests = sum(r['total_tests'] for r in results)
        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0.0
        
        # Calculate pass@k metrics
        pass_at_1 = self.calculate_pass_at_k(results, k=1)
        pass_at_10 = self.calculate_pass_at_k(results, k=10)
        pass_at_100 = self.calculate_pass_at_k(results, k=100)
        
        return {
            'model_name': self.model_name,
            'total_problems': len(results),
            'overall_pass_rate': overall_pass_rate,
            'pass_at_1': pass_at_1,
            'pass_at_10': pass_at_10,
            'pass_at_100': pass_at_100,
            'detailed_results': results
        }
    
    def calculate_pass_at_k(self, results: List[Dict], k: int) -> float:
        """Calculate pass@k metric."""
        pass_rates = [r['pass_rate'] for r in results]
        pass_rates.sort(reverse=True)
        
        if k >= len(pass_rates):
            return np.mean(pass_rates)
        
        return np.mean(pass_rates[:k])
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*50)
        print("HUMANEVAL EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {results['model_name']}")
        print(f"Total Problems: {results['total_problems']}")
        print(f"Overall Pass Rate: {results['overall_pass_rate']:.4f}")
        print(f"Pass@1: {results['pass_at_1']:.4f}")
        print(f"Pass@10: {results['pass_at_10']:.4f}")
        print(f"Pass@100: {results['pass_at_100']:.4f}")
        print("="*50)


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate code generation models on HumanEval")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3-8B", 
                       help="Model name or path")
    parser.add_argument("--use-local", action="store_true", 
                       help="Use local model implementation")
    parser.add_argument("--max-problems", type=int, default=None,
                       help="Maximum number of problems to evaluate")
    parser.add_argument("--output", type=str, default="results/humaneval_results.json",
                       help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = HumanEvalEvaluator(
        model_name=args.model,
        use_local_model=args.use_local
    )
    
    # Run evaluation
    print(f"Starting HumanEval evaluation with model: {args.model}")
    results = evaluator.evaluate_all(max_problems=args.max_problems)
    
    # Print and save results
    evaluator.print_summary(results)
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()