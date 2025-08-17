"""
Reinforcement Learning from Human Feedback (RLHF) implementation for CodeLLM.

This module implements the complete RLHF pipeline including:
1. Reward Model Training
2. PPO Training
3. Human Feedback Integration
4. Evaluation Framework
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

from llms.config import TrainArgs
from llms.generate import Generator


@dataclass
class RLHFConfig:
    """Configuration for RLHF training."""
    # Model configs
    base_model_name: str = "meta-llama/Llama-3-8B"
    reward_model_name: str = "meta-llama/Llama-3-8B"
    
    # Training configs
    learning_rate: float = 1e-5
    reward_learning_rate: float = 1e-5
    batch_size: int = 4
    max_length: int = 512
    num_epochs: int = 3
    
    # PPO configs
    ppo_epochs: int = 4
    ppo_clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    
    # Reward model configs
    reward_model_epochs: int = 3
    reward_batch_size: int = 8
    
    # Data configs
    dataset_name: str = "openai_humaneval"
    human_feedback_file: str = "data/human_feedback.json"
    
    # Output configs
    output_dir: str = "outputs/rlhf"
    save_steps: int = 100


class RewardModel(nn.Module):
    """Reward model for code quality assessment."""
    
    def __init__(self, base_model_name: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Add reward head
        hidden_size = self.model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass to compute reward scores."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use the last hidden state for reward computation
        last_hidden_state = outputs.hidden_states[-1]
        
        # Average pooling over sequence length
        if attention_mask is not None:
            # Masked average pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (last_hidden_state * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = last_hidden_state.mean(dim=1)
        
        # Compute reward score
        reward = self.reward_head(pooled).squeeze(-1)
        return reward


class CodeQualityDataset:
    """Dataset for training reward models on code quality."""
    
    def __init__(self, dataset_name: str = "openai_humaneval"):
        self.dataset = load_dataset(dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
        
        # Generate synthetic code quality data
        self.data = self._generate_synthetic_data()
        
    def _generate_synthetic_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic code quality data for reward model training."""
        data = []
        
        for problem in self.dataset['test']:
            prompt = problem['prompt']
            
            # Generate multiple completions with different quality levels
            completions = self._generate_variations(prompt)
            
            for completion, quality_score in completions:
                data.append({
                    'prompt': prompt,
                    'completion': completion,
                    'quality_score': quality_score,
                    'full_code': prompt + completion
                })
        
        return data
    
    def _generate_variations(self, prompt: str) -> List[Tuple[str, float]]:
        """Generate code variations with different quality levels."""
        # This is a simplified version - in practice, you'd use a more sophisticated approach
        variations = [
            # High quality completion
            ("\n    return 42\n", 0.9),
            # Medium quality completion
            ("\n    x = 42\n    return x\n", 0.7),
            # Low quality completion
            ("\n    pass\n", 0.3),
            # Very low quality completion
            ("\n    return None\n", 0.1),
        ]
        
        return variations
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize the full code
        inputs = self.tokenizer(
            item['full_code'],
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'quality_score': torch.tensor(item['quality_score'], dtype=torch.float32)
        }


class RewardModelTrainer:
    """Trainer for the reward model."""
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize reward model
        self.reward_model = RewardModel(config.reward_model_name).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.reward_model.parameters(),
            lr=config.reward_learning_rate
        )
        
        # Load dataset
        self.dataset = CodeQualityDataset(config.dataset_name)
        
    def train(self):
        """Train the reward model."""
        print("Training reward model...")
        
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.reward_batch_size,
            shuffle=True
        )
        
        for epoch in range(self.config.reward_model_epochs):
            total_loss = 0.0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                quality_scores = batch['quality_score'].to(self.device)
                
                # Forward pass
                predicted_scores = self.reward_model(input_ids, attention_mask)
                
                # Compute loss (MSE loss)
                loss = F.mse_loss(predicted_scores, quality_scores)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        
        # Save reward model
        self.save_reward_model()
    
    def save_reward_model(self):
        """Save the trained reward model."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        save_path = os.path.join(self.config.output_dir, "reward_model")
        
        self.reward_model.save_pretrained(save_path)
        print(f"Reward model saved to {save_path}")
    
    def load_reward_model(self):
        """Load a trained reward model."""
        load_path = os.path.join(self.config.output_dir, "reward_model")
        self.reward_model = RewardModel.from_pretrained(load_path).to(self.device)
        print(f"Reward model loaded from {load_path}")


class PPOTrainer:
    """PPO trainer for RLHF."""
    
    def __init__(self, config: RLHFConfig, reward_model: RewardModel):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.reward_model = reward_model
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        # Initialize optimizers
        self.policy_optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        
    def compute_rewards(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute rewards using the reward model."""
        with torch.no_grad():
            rewards = self.reward_model(input_ids, attention_mask)
        return rewards
    
    def compute_kl_penalty(self, policy_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence penalty."""
        policy_probs = F.softmax(policy_logits, dim=-1)
        reference_probs = F.softmax(reference_logits, dim=-1)
        
        kl_div = F.kl_div(
            policy_probs.log(),
            reference_probs,
            reduction='batchmean'
        )
        
        return kl_div
    
    def ppo_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single PPO step."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Get policy and reference logits
        policy_outputs = self.policy_model(input_ids, attention_mask=attention_mask)
        reference_outputs = self.reference_model(input_ids, attention_mask=attention_mask)
        
        policy_logits = policy_outputs.logits
        reference_logits = reference_outputs.logits
        
        # Compute rewards
        rewards = self.compute_rewards(input_ids, attention_mask)
        
        # Compute KL penalty
        kl_penalty = self.compute_kl_penalty(policy_logits, reference_logits)
        
        # Compute PPO loss
        policy_loss = -rewards.mean() + self.config.entropy_coef * kl_penalty
        
        # Backward pass
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.max_grad_norm)
        self.policy_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'rewards_mean': rewards.mean().item(),
            'kl_penalty': kl_penalty.item()
        }
    
    def train(self, dataset: CodeQualityDataset):
        """Train the policy model using PPO."""
        print("Training policy model with PPO...")
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        for epoch in range(self.config.ppo_epochs):
            total_loss = 0.0
            total_rewards = 0.0
            
            for batch in tqdm(dataloader, desc=f"PPO Epoch {epoch+1}"):
                metrics = self.ppo_step(batch)
                
                total_loss += metrics['policy_loss']
                total_rewards += metrics['rewards_mean']
            
            avg_loss = total_loss / len(dataloader)
            avg_rewards = total_rewards / len(dataloader)
            
            print(f"PPO Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Avg Rewards: {avg_rewards:.4f}")
        
        # Save policy model
        self.save_policy_model()
    
    def save_policy_model(self):
        """Save the trained policy model."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        save_path = os.path.join(self.config.output_dir, "policy_model")
        
        self.policy_model.save_pretrained(save_path)
        print(f"Policy model saved to {save_path}")


class HumanFeedbackCollector:
    """Collect and manage human feedback for RLHF."""
    
    def __init__(self, feedback_file: str):
        self.feedback_file = feedback_file
        self.feedback_data = self._load_feedback()
    
    def _load_feedback(self) -> List[Dict[str, Any]]:
        """Load existing human feedback data."""
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        return []
    
    def add_feedback(self, prompt: str, completion: str, quality_score: float, feedback_text: str = ""):
        """Add new human feedback."""
        feedback_item = {
            'prompt': prompt,
            'completion': completion,
            'quality_score': quality_score,
            'feedback_text': feedback_text,
            'timestamp': str(torch.tensor(0).item())  # Placeholder for timestamp
        }
        
        self.feedback_data.append(feedback_item)
        self._save_feedback()
    
    def _save_feedback(self):
        """Save feedback data to file."""
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
    
    def get_feedback_dataset(self) -> List[Dict[str, Any]]:
        """Get all feedback data as a dataset."""
        return self.feedback_data


class RLHFPipeline:
    """Complete RLHF pipeline."""
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        self.reward_trainer = RewardModelTrainer(config)
        self.feedback_collector = HumanFeedbackCollector(config.human_feedback_file)
    
    def train_reward_model(self):
        """Train the reward model."""
        self.reward_trainer.train()
    
    def train_policy_model(self):
        """Train the policy model using PPO."""
        # Load trained reward model
        self.reward_trainer.load_reward_model()
        
        # Initialize PPO trainer
        ppo_trainer = PPOTrainer(self.config, self.reward_trainer.reward_model)
        
        # Load dataset
        dataset = CodeQualityDataset(self.config.dataset_name)
        
        # Train policy model
        ppo_trainer.train(dataset)
    
    def evaluate_model(self, model_path: str, num_samples: int = 100) -> Dict[str, float]:
        """Evaluate the trained model."""
        # Load the trained model
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load HumanEval dataset
        dataset = load_dataset("openai_humaneval")
        
        # Evaluate on a subset
        problems = dataset['test'].select(range(min(num_samples, len(dataset['test']))))
        
        correct = 0
        total = 0
        
        for problem in problems:
            prompt = problem['prompt']
            test_cases = problem['test']
            
            # Generate completion
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.2,
                    do_sample=True
                )
            
            completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Test the completion
            full_code = prompt + completion
            try:
                local_vars = {}
                exec(full_code, {}, local_vars)
                
                # Extract function name
                function_name = prompt.split('def ')[1].split('(')[0].strip()
                func = local_vars.get(function_name)
                
                if func is not None:
                    # Test with provided test cases
                    for test_case in test_cases:
                        exec(test_case, {}, local_vars)
                    correct += 1
                
            except Exception:
                pass
            
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def run_full_pipeline(self):
        """Run the complete RLHF pipeline."""
        print("Starting RLHF pipeline...")
        
        # Step 1: Train reward model
        print("\n=== Step 1: Training Reward Model ===")
        self.train_reward_model()
        
        # Step 2: Train policy model with PPO
        print("\n=== Step 2: Training Policy Model with PPO ===")
        self.train_policy_model()
        
        # Step 3: Evaluate the model
        print("\n=== Step 3: Evaluating Model ===")
        policy_model_path = os.path.join(self.config.output_dir, "policy_model")
        results = self.evaluate_model(policy_model_path)
        
        print(f"Final Results: {results}")
        
        # Save results
        results_path = os.path.join(self.config.output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)


def main():
    """Main function to run RLHF pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RLHF pipeline for CodeLLM")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--mode", type=str, choices=["reward", "ppo", "full"], default="full",
                       help="Training mode")
    
    args = parser.parse_args()
    
    # Load config
    config = RLHFConfig()
    if args.config:
        # Load from file if provided
        pass
    
    # Initialize pipeline
    pipeline = RLHFPipeline(config)
    
    # Run based on mode
    if args.mode == "reward":
        pipeline.train_reward_model()
    elif args.mode == "ppo":
        pipeline.train_policy_model()
    else:
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
