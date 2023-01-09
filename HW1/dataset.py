from typing import List, Dict
from torch.utils.data import Dataset
from utils import Vocab, pad_to_len
import torch
# import seqeval

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        ret = {}
        ret['id'] = [s['id'] for s in samples]
        ret['text'] = [s['text'].split() for s in samples]
        to_len = max(len(s) for s in ret['text'])
        to_len = min(to_len, self.max_len)
        ret['text'] = torch.LongTensor(self.vocab.encode_batch(ret['text'], to_len))
        
        if 'intent' in samples[0].keys():
            ret['intent'] = [self.label2idx(s['intent']) for s in samples]
            ret['intent'] = torch.LongTensor(ret['intent'])
        return ret

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]



class SeqTaggingClsDataset(SeqClsDataset):
    def collate_fn(self, samples):
        # TODO: implement collate_fn
        ret = {}
        ret['id'] = [s['id'] for s in samples]
        ret['tokens'] = [s['tokens'] for s in samples]
        ret['len'] = [len(s['tokens']) for s in samples]
        to_len = max(ret['len'])

        ret['tokens'] = self.vocab.encode_batch(ret['tokens'], to_len)
        ret['tokens'] = torch.LongTensor(ret['tokens'])
        if 'tags' in samples[0].keys():
            ret['tags'] = [[self.label2idx(x) for x in s['tags']] for s in samples]
            ret['tags'] = pad_to_len(ret['tags'], to_len, self.label2idx("O")) # Use "O" to pad
            ret['tags'] = torch.LongTensor(ret['tags'])

        return ret

    def decode_labels(self, data: List[List[int]], len_info: List[int]=None) -> List[List[str]]:
        # print([[self.idx2label(x) for x in s] for s in data])
        data = [[self.idx2label(x) for x in s] for s in data]
        return data if len_info is None else self.remove_pads(data, len_info)
    
    def remove_pads(self, data: List[List[int]], len_info: List[int]):
        for i in range(len(len_info)):
            data[i] = data[i][:len_info[i]]
        return data

