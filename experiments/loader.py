import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import argparse
import logging

from collections import defaultdict

def collate_hred(batch_list, vectorizer):
    logging.debug(batch_list)
    batch_dict = defaultdict(list)
    ys = [item['y_target'] for item in batch_list]
    ylens = torch.Tensor([len(y) for y in ys])
    ys_padded = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=vectorizer.pad_idx)

    for item in batch_list:
        batch_dict['x_data'].append(item['x_data'])
    batch_dict['y_target'] = ys_padded
    return batch_dict


def prepare_hred_dataloader(datasets, vectorizer, args):
    train_loader = DataLoader(datasets.train, batch_size=args.train_batch_size, collate_fn=lambda batch: collate_hred(batch, vectorizer))
    val_loader = DataLoader(datasets.val, batch_size=args.val_batch_size, collate_fn=lambda batch: collate_hred(batch, vectorizer))
    test_loader = DataLoader(datasets.test, batch_size=args.test_batch_size, collate_fn=lambda batch: collate_hred(batch, vectorizer))

    loaders = argparse.Namespace(
        train=train_loader,
        val=val_loader,
        test=test_loader)

    return loaders