from glob import glob

# very slow
# from bpe import Tokenizer
# tokenizer = Tokenizer()
# files = glob('/Users/shijiewu/Code/commitpackft/data/*/*jsonl')
# long_text = []
# for path in files:
#     with open(path) as f:
#         long_text.append(''.join(f.readlines()))
# long_text = ''.join(long_text)
# tokenizer.train(long_text, 256 + 100)
# path = tokenizer.save()
# print(f'trained tokenizer saved in {path}')

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

paths = glob('/Users/shijiewu/Code/commitpackft/data/*/*jsonl')

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=4_256, min_frequency=2)

# Save files to disk
tokenizer.save_model(".", "hf_commit")
