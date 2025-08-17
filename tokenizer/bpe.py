"""
Reference: https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py
"""
import os
import regex as re

from collections import defaultdict

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def get_stat(bytes, count=None):
    count = defaultdict(int) if count is None else count
    for pair in zip(bytes, bytes[1:]):
        count[pair] += 1
    return count

def merge(ids, pair, new_idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i + 1 < len(ids) and ids[i + 1] == pair[1]:
            new_ids.append(new_idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

class Tokenizer:
    def __init__(self, prefix='tinycode', pattern=None):
        self.prefix = prefix
        self.pattern = GPT2_SPLIT_PATTERN if pattern is None or pattern == 'gpt4' else GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.raw_vocab = None
        self.vocab = {}
        self.special_tokens = {}
        self.inverse_special_tokens = {}
    
    @staticmethod
    def load_from_file(path):
        with open(path, 'r') as f:
            lines = [line.rstrip().split() for line in f.readlines()]
        pattern = lines[0][1]
        num_special = int(lines[1][0])
        tokenizer = Tokenizer(pattern=pattern)
        tokenizer.register_special({line[0]: int(line[1]) for line in lines[2:2 + num_special]})
        # (idx, idx) -> idx
        tokenizer.vocab = {tuple([int(n) for n in pair]): 256 + i for i, pair in enumerate(lines[2 + num_special:])}
        return tokenizer
    
    @staticmethod
    def load_from_hf(path, json_path):
        def to_bytes(c):
            ans = ord(c)
            if ans >= 256: ans -= 256
            return ans
        
        import json
        with open(path, 'r') as f:
            # (str(b), str(b))
            lines = [line.rstrip().split() for line in f.readlines()]
        with open(json_path, 'r') as f:
            # str(b) -> idx
            vocab = json.load(f)
        
        real_vocab = {}
        for line in lines[1:]:
            real_vocab[vocab[line[0]], vocab[line[1]]] = vocab[''.join(line)]
        tokenizer = Tokenizer()
        tokenizer.vocab = real_vocab
        tokenizer.raw_vocab = {to_bytes(k[0]): v for k, v in vocab.items() if v in set(range(256))}
        return tokenizer

    def train(self, text, vocab_size):
        """
        Learn vocabulary from text
        """
        text_chunks = re.findall(self.compiled_pattern, text)
        word_bytes = [list(ch.encode('utf-8')) for ch in text_chunks]
        merges = vocab_size - 256
        for i in range(merges):
            count = defaultdict(int)
            # find most frequent pairs
            for word_byte in word_bytes:
                get_stat(word_byte, count)
            pair = max(count, key=count.get)
            new_idx = 256 + i
            self.vocab[pair] = new_idx
            # merge pair and create new word bytes
            new_word_bytes = []
            for word_byte in word_bytes:
                new_word_byte = merge(word_byte, pair, new_idx)
                new_word_bytes.append(new_word_byte)
            word_bytes = new_word_bytes
    
    def save(self):
        with open(self._save_path(), 'w+') as f:
            f.write(f'pattern {self.pattern}\n')
            f.write(f'{len(self.special_tokens)}\n')
            for special, idx in self.special_tokens.items():
                f.write(f'{special} {idx}\n')
            for word1, word2 in self.vocab:
                f.write(f'{word1} {word2}\n')
        return self._save_path()
    
    def _save_path(self):
        return os.path.join(f'{self.prefix}.model')
    
    def encode_ordinary(self, text):
        text_chunks = re.findall(self.pattern, text)
        idxs = []
        for chunk in text_chunks:
            bytes = chunk.encode('utf-8')
            while len(bytes) >= 2:
                count = get_stat(bytes)
                # lambda maps from pair to priority
                pair = min(count, key=lambda p: self.vocab.get(p, float('inf')))
                if pair not in self.vocab:
                    break
                new_bytes = merge(bytes, pair, self.vocab[pair])
                bytes = new_bytes
            idxs.extend(bytes)
        return idxs
    
    def register_special(self, special_tokens):
        # example: str -> int
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def encode(self, text):
        if not self.special_tokens:
            return self.encode_ordinary(text)
        special_pattern = '(' + '|'.join([re.escape(s) for s in self.special_tokens]) + ')'
        chunks = re.split(special_pattern, text)
        ids = []
        for part in chunks:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def _prepare_id2pair(self):
        if not getattr(self, 'id2pair', None):
            self.id2pair = {idx: bytes([idx]) for idx in range(256)}
            for (p0, p1), idx in self.vocab.items():
                self.id2pair[idx] = self.id2pair[p0] + self.id2pair[p1]

    def decode(self, idxs):
        # int -> string
        self._prepare_id2pair()
        bytes = []
        for idx in idxs:
            if idx in self.id2pair:
                bytes.append(self.id2pair[idx])
            elif idx in self.inverse_special_tokens:
                bytes.append(self.inverse_special_tokens[idx].encode('utf-8'))

        text_bytes = b''.join(bytes)
        text = text_bytes.decode('utf-8', errors='replace')
        return text