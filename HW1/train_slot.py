import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab
from seqeval.metrics import accuracy_score, classification_report, f1_score
from seqeval.scheme import IOB2

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def tokens_compare(tags1: List[List[str]], tags2: List[List[str]]):
    return sum([tags1[i][j] == tags2[i][j] for i in range(len(tags1)) for j in range(len(tags1[i]))]), sum([len(s) for s in tags1])

def sentence_compare(tags1: List[List[str]], tags2: List[List[str]]):
    return sum([tags1[i] == tags2[i] for i in range(len(tags1))]), len(tags1)

def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    # TODO: create DataLoader for train / dev datasets
    dataloaders: Dict[str, torch.utils.data.DataLoader] = {
        split: torch.utils.data.DataLoader(
            dataset=datasets[split],
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=datasets[split].collate_fn,
            num_workers=8 # Recommended: num of CPUs
        )
        for split, split_data in data.items()
    }
    t_batch = len(dataloaders['train']) 
    v_batch = len(dataloaders['eval'])

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqTagger(embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional, num_class=datasets['train'].num_classes)
    model = model.to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.8)
    criterion = torch.nn.BCELoss()

    total_loss, best_acc = 0, 0


    for epoch in trange(args.num_epoch, desc="Epoch"):
        total_loss, total_acc, total_num = 0, 0, 0
        sen_acc, sen_num = 0, 0
        pred_tags, real_tags = [], []
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        for i, batch in enumerate(dataloaders['train']):
            batch['tokens'] = batch['tokens'].to(args.device)
            batch['tags'] = batch['tags'].to(args.device)
            optimizer.zero_grad()
            outputs = model(batch)

            y = torch.nn.functional.one_hot(batch['tags'], model.num_class).float()
            loss = criterion(outputs['pred_logits'], y) # loss for each estimation of num_class
            loss.backward()
            optimizer.step()
            
            # print('pred_labels:', outputs['pred_labels'].shape)

            total_loss += loss.item()
            # print("total_loss:", total_loss)
            
            tags1 = datasets[TRAIN].decode_labels(outputs['pred_labels'].tolist(), batch['len'])
            tags2 = datasets[TRAIN].decode_labels(batch['tags'].tolist(), batch['len'])
            pred_tags.extend(tags1)
            real_tags.extend(tags2)
            acc, num = tokens_compare(tags1, tags2)
            total_acc += acc
            total_num += num
            s_acc, s_num = sentence_compare(tags1, tags2)
            sen_acc += s_acc
            sen_num += s_num

            print('[ Epoch{}: {}/{} ] loss:{:.6f} token:{:.3f} sentence:{:.3f}'.format(
                epoch + 1, i + 1, t_batch, loss.item(), acc * 100 / num, sen_acc * 100 / sen_num),end='\r')

        print('\nTrain | Loss:{:.5f} token: {:.3f} sentence: {:.3f}'.format(
            total_loss / t_batch,
            total_acc * 100 / total_num, 
            sen_acc * 100 / sen_num, 
        ))

        scheduler.step()

        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        with torch.no_grad():
            total_loss, total_acc, total_num = 0, 0, 0
            sen_acc, sen_num = 0, 0
            pred_tags, real_tags = [], []
            for i, batch in enumerate(dataloaders['eval']):


                batch['tokens'] = batch['tokens'].to(args.device)
                batch['tags'] = batch['tags'].to(args.device)

                outputs = model(batch)
                y = torch.nn.functional.one_hot(batch['tags'], model.num_class).float()
                loss = criterion(outputs['pred_logits'], y)


                total_loss += loss.item()

                tags1 = datasets[TRAIN].decode_labels(outputs['pred_labels'].tolist(), batch['len'])
                tags2 = datasets[TRAIN].decode_labels(batch['tags'].tolist(), batch['len'])
                pred_tags.extend(tags1)
                real_tags.extend(tags2)
                acc, num = tokens_compare(tags1, tags2)
                total_acc += acc
                total_num += num
                s_acc, s_num = sentence_compare(tags1, tags2)
                sen_acc += s_acc
                sen_num += s_num

            print('\nVal | Loss:{:.5f} token: {:.3f} sentence: {:.3f}'.format(
                total_loss / t_batch,
                total_acc * 100 / total_num, 
                sen_acc * 100 / sen_num, 
            ))
            if sen_acc / sen_num > best_acc:
                best_acc = sen_acc / sen_num
                torch.save(model.state_dict(), "{}/model.pt".format(args.ckpt_dir))
                print('saving model with acc {:.3f}'.format(best_acc * 100))
            
            report = classification_report(
                real_tags,
                pred_tags, 
                scheme=IOB2,
                mode='strict'
            )
        print(report)
    print("best acc:{:.3f}".format(best_acc * 100))
    
    # TODO: Inference on test set
    
    """
    print('Inference on test set')
    for _, batch in enumerate(dataloaders['eval']):
        print('len(data):', len(data))
        batch['text'] = batch['text'].to(args.device)
        batch['intent'] = batch['intent'].to(args.device)
        
        optimizer.zero_grad()
        outputs = model(batch)

        loss = outputs['loss']
        loss.backward()
        optimizer.step()

        correct = torch.sum(torch.eq(outputs['pred_labels'], batch['intent'])).item()
        total_acc += (correct / args.batch_size)
        total_loss += loss.item()
        print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f}'.format(
        epoch + 1, i + 1, t_batch, loss.item(), correct * 100 / args.batch_size),end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss / t_batch, total_acc / t_batch * 100))
    """


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default='cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)

