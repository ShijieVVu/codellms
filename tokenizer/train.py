"""
Tokenizer training script for CodeLLM.

This script demonstrates how to train a BPE (Byte Pair Encoding) tokenizer from scratch
on code data. It provides two approaches:
1. Custom BPE implementation (slower but educational)
2. HuggingFace tokenizers library (faster, production-ready)
"""

from glob import glob
import os
from pathlib import Path

# Option 1: Custom BPE implementation (educational)
def train_custom_bpe():
    """Train using custom BPE implementation - slower but educational."""
    print("Training custom BPE tokenizer...")
    
    # Uncomment to use custom implementation
    # from bpe import Tokenizer
    # tokenizer = Tokenizer()
    # 
    # # Load code files
    # files = glob('/path/to/code/data/*/*.py')  # Adjust path as needed
    # long_text = []
    # for path in files:
    #     with open(path) as f:
    #         long_text.append(''.join(f.readlines()))
    # long_text = ''.join(long_text)
    # 
    # # Train tokenizer
    # tokenizer.train(long_text, 50257)  # GPT-2 vocabulary size
    # path = tokenizer.save()
    # print(f'Custom BPE tokenizer saved to {path}')

# Option 2: HuggingFace tokenizers (recommended)
def train_hf_tokenizer():
    """Train using HuggingFace tokenizers library - faster and production-ready."""
    print("Training HuggingFace tokenizer...")
    
    try:
        from tokenizers import ByteLevelBPETokenizer
        
        # Configure data paths - adjust these for your code dataset
        data_paths = [
            # Example paths - modify for your dataset
            # '/path/to/code/data/python/*.py',
            # '/path/to/code/data/javascript/*.js',
            # '/path/to/code/data/java/*.java',
        ]
        
        # If no paths configured, use a sample
        if not data_paths or not any(os.path.exists(p) for p in data_paths):
            print("No valid data paths found. Using sample data...")
            # Create sample code data for demonstration
            sample_code = '''
def hello_world():
    print("Hello, World!")
    return True

class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b

# Main execution
if __name__ == "__main__":
    calc = Calculator()
    result = calc.add(5, 3)
    print(f"5 + 3 = {result}")
'''
            
            # Save sample to temporary file
            os.makedirs("temp_data", exist_ok=True)
            with open("temp_data/sample.py", "w") as f:
                f.write(sample_code)
            
            # Create additional sample files for better training
            sample_js = '''
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

class ArrayUtils {
    static sum(arr) {
        return arr.reduce((a, b) => a + b, 0);
    }
    
    static filter(arr, predicate) {
        return arr.filter(predicate);
    }
}
'''
            
            sample_java = '''
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
    
    public static int factorial(int n) {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    }
}
'''
            
            with open("temp_data/sample.js", "w") as f:
                f.write(sample_js)
            with open("temp_data/sample.java", "w") as f:
                f.write(sample_java)
            
            data_paths = ["temp_data/sample.py", "temp_data/sample.js", "temp_data/sample.java"]
        
        # Initialize tokenizer
        tokenizer = ByteLevelBPETokenizer()
        
        # Customize training parameters
        vocab_size = 50257  # GPT-2 vocabulary size
        min_frequency = 2   # Minimum frequency for a token to be included
        
        print(f"Training tokenizer with vocab_size={vocab_size}, min_frequency={min_frequency}")
        
        # Train the tokenizer
        tokenizer.train(
            files=data_paths, 
            vocab_size=vocab_size, 
            min_frequency=min_frequency,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        )
        
        # Save the trained tokenizer
        output_dir = "trained_tokenizer"
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save_model(output_dir, "code_bpe")
        
        print(f"Tokenizer saved to {output_dir}/")
        print(f"Files created:")
        print(f"  - {output_dir}/code_bpe-vocab.json")
        print(f"  - {output_dir}/code_bpe-merges.txt")
        
        # Test the tokenizer
        test_text = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded.ids)
        
        print(f"\nTest encoding:")
        print(f"Input: {test_text}")
        print(f"Tokens: {encoded.tokens}")
        print(f"Decoded: {decoded}")
        
        return output_dir
        
    except ImportError:
        print("HuggingFace tokenizers not installed. Install with: pip install tokenizers")
        return None

def main():
    """Main function to train tokenizer."""
    print("CodeLLM Tokenizer Training")
    print("=" * 40)
    
    # Train using HuggingFace tokenizers (recommended)
    output_dir = train_hf_tokenizer()
    
    if output_dir:
        print(f"\n✅ Tokenizer training completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"\nTo use this tokenizer in your model:")
        print(f"1. Copy the files to your model directory")
        print(f"2. Update config.py with the tokenizer path")
        print(f"3. Use with SentencePieceProcessor or HuggingFace tokenizers")
    else:
        print("\n❌ Tokenizer training failed. Check the error messages above.")

if __name__ == "__main__":
    main()
