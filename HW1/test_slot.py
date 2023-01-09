import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=8 # Recommended: num of CPU
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)
    model = model.to(args.device)

    ids, labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch['tokens'] = batch['tokens'].to(args.device)
            outputs = model(batch)
            labels += dataset.decode_labels(outputs['pred_labels'].tolist(), batch['len'])
            ids += batch['id']


    print("save csv...")
    with open(args.pred_file, 'w') as f:
        f.write('id,tags\n')
        for id, label in zip(ids, labels):
            f.write("%s," % id)
            for i in range(len(label)):
                f.write("%s%c" % (label[i], " \n"[i+1==len(label)]))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default='cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

# CUDA_VISIBLE_DEVICES='7' sh slot_tag.sh data/slot/test.json data/slot/pred3.csv