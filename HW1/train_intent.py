import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
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
    model = SeqClassifier(embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional, num_class=datasets['train'].num_classes)
    model = model.to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    criterion = torch.nn.BCELoss()
    total_loss, total_acc, best_acc = 0, 0, 0


    for epoch in trange(args.num_epoch, desc="Epoch"):
        total_loss, total_acc = 0, 0
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        for i, batch in enumerate(dataloaders['train']):
            # print('len(data):', len(data))
            batch['text'] = batch['text'].to(args.device)
            batch['intent'] = batch['intent'].to(args.device)
            
            optimizer.zero_grad()
            outputs = model(batch)

            # print(outputs['pred_logits'].shape, outputs['pred_labels'].shape)
            y = torch.nn.functional.one_hot(batch['intent'], model.num_class).float()

            # print(outputs['pred_logits'].shape, y.shape)
            loss = criterion(outputs['pred_logits'], y)
            loss.backward()
            optimizer.step()

            correct = torch.sum(torch.eq(outputs['pred_labels'], batch['intent'])).item()
            total_acc += (correct / args.batch_size)
            total_loss += loss.item()
            # print("total_loss:", total_loss)
            print('[ Epoch{}: {}/{} ] loss:{:.6f} acc:{:.3f}'.format(
                epoch + 1, i + 1, t_batch, loss.item(), correct * 100 / args.batch_size),end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f} {}/{}'.format(total_loss / t_batch, total_acc / t_batch * 100, correct, t_batch))
        scheduler.step()

        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, batch in enumerate(dataloaders['eval']):
                batch['text'] = batch['text'].to(args.device)
                batch['intent'] = batch['intent'].to(args.device)

                outputs = model(batch)
                y = torch.nn.functional.one_hot(batch['intent'], model.num_class).float()
                loss = criterion(outputs['pred_logits'], y)
                correct = torch.sum(torch.eq(outputs['pred_labels'], batch['intent'])).item()
                total_acc += (correct / args.batch_size)
                total_loss += loss.item()
            print("Valid | Loss:{:.5f} Acc: {:.3f}".format(total_loss / v_batch, total_acc / v_batch * 100))
            if total_acc > best_acc:
                best_acc = total_acc
                torch.save(model.state_dict(), "{}/model.pt".format(args.ckpt_dir))
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
    print("best acc:{:.3f}".format(best_acc / v_batch * 100))
    
    # TODO: Inference on test set
    
    """
    print('Inference on test set')
    model.train()
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
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
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
