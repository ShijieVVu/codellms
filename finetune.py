# finetune a peft version of model
import os
import torch
from llms.config import TrainArgs as Args, CheckpointMode
from llms.llama import Transformer

from sentencepiece import SentencePieceProcessor
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

from random import randint
import numpy as np

import torch.nn.functional as F
from peft import LoraModel, LoraConfig

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
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.cuda.set_device(0)
    model: Transformer = Transformer(args)
    # if os.path.exists(args.checkpoint_path):
    #     state_dict = torch.load(args.checkpoint_path, map_location='cpu')
    #     if args.checkpoint_mode == CheckpointMode.llama:
    #         llama_2_local = {
    #             'tok_embeddings': 'embedding',
    #             'attention_norm': 'norm1',
    #             'attention': 'attn',
    #             'feed_forward': 'ffn', 
    #             'ffn_norm': 'norm2', 
    #         }
    #         new_state_dict = {}
    #         for key, val in state_dict.items():
    #             for orig, new_key in llama_2_local.items():
    #                 if orig in key:
    #                     key = key.replace(orig, new_key)
    #                     break
    #             new_state_dict[key] = val
    #         state_dict = new_state_dict
    #         del state_dict['rope.freqs']

    #     model.load_state_dict(state_dict, strict=False)
    #     print('loaded pretrained weights')
    lora_config = LoraConfig(r=4, lora_alpha=16, target_modules=['wq', 'wk'], lora_dropout=0.1, bias="none")
    # model = LoraModel(model, lora_config, "default")
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = F.cross_entropy
    
    dataset = TokenizeDataset(args)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size)
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(args.epochs):
            for sample in train_dataloader:
                inputs = sample[:, :-1]
                gt = sample[:, 1:]
                outputs = model.forward_train(inputs)
                loss = loss_fn(outputs.transpose(1, 2), gt)
                optimizer.zero_grad()
                print('loss', loss)
                loss.backward(retain_graph=True)
                optimizer.step()
        
        torch.save(model.parameters(), os.path.join(args.save_folder, 'checkpoint.pth'))

if __name__ == '__main__':
    args = Args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)