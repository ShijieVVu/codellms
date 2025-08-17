import torch
from llms.config import TrainArgs as Args
from llms.generate import Generator


def main():
    args = Args()
    generator = Generator(args)

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        """\
def fizzbuzz(n: int):""",
        """\
import argparse

def main(string: str):
    print(string)
    print(string[::-1])

if __name__ == "__main__":"""
    ]
    # prompts = [
    #     '''from typing import List def has_close_elements(numbers: List[float], threshold: float) -> bool: """ Check if in given list of numbers, are any two numbers closer to each other than given threshold. >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True """''',
    #     '''from typing import List def separate_paren_groups(paren_string: str) -> List[str]: """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to separate those group into separate strings and return the list of those. Separate groups are balanced (each open brace is properly closed) and not nested within each other Ignore any spaces in the input string. >>> separate_paren_groups('( ) (( )) (( )( ))') ['()', '(())', '(()())'] """''',
    # ]
    outputs = generator.text_completion(prompts, temperature=0.0)

    for prompt, output in zip(prompts, outputs):
        print(prompt)
        print(f"> {output}")
        print("\n==================================\n")


if __name__ == '__main__':
    main()