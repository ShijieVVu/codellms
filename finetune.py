# finetune a peft version of model
import os
import torch
from llms.config import TrainArgs as Args
from llms.llama import Transformer

from sentencepiece import SentencePieceProcessor
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

from random import randint
import numpy as np

import torch.nn.functional as F


class TokenizeDataset(IterableDataset):
    def __init__(self, args: Args):
        self.stream_dataset = load_dataset(args.dataset, streaming=True, split="train", token=args.hf_token)
        self.sp_model = SentencePieceProcessor(model_file=args.tokenizer_model)
        self.max_length = args.max_length
    
    def __iter__(self):
        for item in self.stream_dataset:
            token_list = self.sp_model.encode(item['content'])
            if self.max_length > len(token_list):
                token_list.extend([self.sp_model.eos_id()] * (self.max_length - len(token_list)))
            else:
                start_index = randint(0, len(token_list) - self.max_length)
                token_list = token_list[start_index: start_index + self.max_length]
            token_list = np.array(token_list)
            yield token_list


def main(args: Args):
    model: Transformer = Transformer(args)
    model_path = os.path.join(args.output_dir, 'model.ckpt')
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    model.to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = F.cross_entropy
    
    dataset = TokenizeDataset(args)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size)
    for epoch in range(args.epochs):
        for sample in train_dataloader:
            sample = sample.to('cuda')
            inputs = sample[:, :-1]
            gt = sample[:, 1:]
            outputs = model(inputs, 0)
            loss = loss_fn(outputs.transpose(1, 2), gt)
            
            optimizer.zero_grad()
            print('loss', loss)
            loss.backward()
            optimizer.step()
        
        torch.save(model.parameters(), model_path)

if __name__ == '__main__':
    args = Args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)