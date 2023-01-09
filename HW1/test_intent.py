import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    
    # TODO: create DataLoader for test dataset
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=8 # Recommended: num of CPU
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
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

    # TODO: predict dataset
    ids, labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch['text'] = batch['text'].to(args.device)
            outputs = model(batch)
            labels += outputs['pred_labels'].tolist()
            ids += batch['id']


    # TODO: write prediction to file (args.pred_file)
    print("save csv...")
    with open(args.pred_file, 'w') as f:
        f.write('id,intent\n')
        for id, label in zip(ids, labels):
            f.write("%s,%s\n" % (id, dataset.idx2label(label)))


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
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
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

# sh intent_cls.sh data/slot/train.json data/slot/pred.csv