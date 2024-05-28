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
    outputs = generator.text_completion(prompts)
    for prompt, output in zip(prompts, outputs):
        print(prompt)
        print(f"> {output}")
        print("\n==================================\n")


if __name__ == '__main__':
    main()