import time
import torch
from typing import List

from llms.config import TrainArgs as Args, CheckpointMode
from llms.llama import Transformer
from sentencepiece import SentencePieceProcessor

import torch.nn.functional as F


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class Generator:
    def __init__(self, args: Args):
        self.max_length = args.max_length
        start_time = time.time()
        state_dict = torch.load(args.checkpoint_path, map_location='cpu')
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        torch.cuda.set_device(0)
        self.transformer = Transformer(args)

        if args.checkpoint_mode == CheckpointMode.llama:
            llama_2_local = {
                'tok_embeddings': 'embedding',
                'attention_norm': 'norm1',
                'attention': 'attn',
                'feed_forward': 'ffn', 
                'ffn_norm': 'norm2', 
            }
            new_state_dict = {}
            for key, val in state_dict.items():
                for orig, new_key in llama_2_local.items():
                    if orig in key:
                        key = key.replace(orig, new_key)
                        break
                new_state_dict[key] = val
            state_dict = new_state_dict
            del state_dict['rope.freqs']
        self.transformer.load_state_dict(state_dict, strict=False)
        self.spm = SentencePieceProcessor(args.tokenizer_model)

        print(f"Loaded in {time.time() - start_time:.2f} seconds")

    def text_completion(self, 
                        prompts: List[str],
                        temperature: float = 0.6,
                        top_p: float = 0.9) -> List[str]:
        """
        input is list of prompt strings, output is list of completed strings
        """
        tokens = [[self.spm.bos_id()] + self.spm.encode(p) for p in prompts]
        prompt_lengths = [len(t) for t in tokens]
        min_prompt_length = min(prompt_lengths)
        bsz = len(prompts)
        memory = torch.full((bsz, self.max_length), fill_value=self.spm.pad_id(), dtype=torch.int32)
        for b, t in enumerate(tokens):
            memory[b, :len(t)] = torch.cuda.IntTensor(t)
        prev_pos = 0
        over = torch.tensor([False] * bsz)
        for cur_pos in range(min_prompt_length, self.max_length):
            logits = self.transformer(memory[:, prev_pos: cur_pos], prev_pos)
            next_logits = logits[:, -1]
            if temperature == 0:
                next_tokens = next_logits.argmax(dim=-1)
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_tokens = sample_top_p(probs, top_p)[:, 0]
            memory[:, cur_pos] = torch.where(memory[:, cur_pos] == self.spm.pad_id(), next_tokens, memory[:, cur_pos])
            over |= memory[:, cur_pos] == self.spm.eos_id()
            if all(over):
                break
            prev_pos = cur_pos
        
        response = []
        for b, tokens in enumerate(memory):
            tokens = tokens.tolist()
            end = tokens.index(self.spm.eos_id(), prompt_lengths[b]) if self.spm.eos_id() in tokens else len(tokens)
            response.append(self.spm.decode(tokens[prompt_lengths[b]:end]))

        return response