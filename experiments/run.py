import argparse
import torch
import torch.optim as optim
import torch.nn as nn

from dataset import HredDataset, MixedShortDataset, MemNetDataset
import os
import json
import logging
import pickle as pkl

from vocabulary import Vocabulary
from vectorizer import SequenceVectorizer
from loader import prepare_hred_dataloader

import train_model

import models.hred as hred

# logging.basicConfig(level=logging.DEBUG)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_bidaf_train_data(args):
    path = os.path.join(args.data_dir,
        'experiment_data',
        'bidaf',
        'mixed_short',
        'train-v1.1.json')
    with open(path, 'r') as train_file:
        data = json.load(train_file)
    num_examples = len(data["data"])
    chats = data["data"]
    logging.debug("Loaded {} examples".format(num_examples))

    # Prepare the training data 
    train_dataset = MixedShortDataset(chats)

    return train_dataset

def load_hred_dataset(args, filename, vectorizer):
    path = os.path.join(args.data_dir,
        'experiment_data',
        'hred',
        filename)

    with open(path, 'r') as train_file:
        data = json.load(train_file)

    contexts = data[0]
    responses = data[1]
    logging.debug("Loaded {} examples".format(len(responses)))
    dataset = HredDataset(contexts, responses, vectorizer, args.device)

    return dataset

def load_memnet_train_data(args, vectorizer):
    path = os.path.join(args.data_dir,
        'memnet_data',
        'train.json')

    with open(path, 'r') as train_file:
        data = json.load(train_file)

    dataset = MemNetDataset(data, vectorizer)

    return dataset


def load_hred_vocabulary(args):
    path = os.path.join(args.data_dir,
        'hred',
        'words.pkl')

    with open(path, 'rb') as words_file:
        words = pkl.load(words_file)
    word_dict = {word: idx for idx, word in enumerate(words.keys())}
    logging.debug(type(words))
    vocab = Vocabulary.from_dict(word_dict)

    return vocab

def load_memnet_vocabulary(args):
    vocab_path = os.path.join(args.data_dir,
        'memnet_data',
        'vocab.pkl')
    with open(vocab_path, 'rb') as vocab_file:
        vocab = pkl.load(vocab_file)

    return vocab

def create_hred_model(args, vocab):
    word_encoder = hred.WordEncoder(
        input_size=len(vocab),
        embed_size=args.embed_size,
        hidden_size=args.word_hidden_size,
        bidirectional=args.bidirectional)
    context_encoder = hred.ContextEncoder(word_encoder, args.context_hidden_size, args.device)
    decoder = hred.Decoder(
        output_size=len(vocab),
        embed_size=args.embed_size,
        hidden_size=args.decoder_hidden_size)

    seq2seq = hred.Seq2Seq(context_encoder, decoder, args.device)
    
    return seq2seq

def create_memnet_model(args, in_vocab, out_vocab):
    pass


def base_parser():
    parser = argparse.ArgumentParser(
        description="Run the experiments for the models with knowledge on the Holl-E dataset")
    parser.add_argument("--data_dir",
        default="holle/")
    parser.add_argument("--n_epochs", default=20, 
        type=int, help="Number of epochs to train the model for")
    parser.add_argument('--learning_rate', default=0.5, 
        type=float, help='Learning rate for the model')
    parser.add_argument('--train_batch_size', default=16, 
        type=int, help='Batch size for train')
    parser.add_argument('--val_batch_size', default=16, 
        type=int, help='Batch size for validation')
    parser.add_argument('--test_batch_size', default=16, 
        type=int, help='Batch size for test')
    parser.add_argument('--model', default="hred",
        choices=['hred'], help='Model to train')

    return parser

def hred_parser():
    parser = argparse.ArgumentParser(
        description="Parameters for hred")
    parser.add_argument("--word_hidden_size", default=100,
        type=int, help="Hidden size of word encoder")
    parser.add_argument("--context_hidden_size", default=100,
        type=int, help="Hidden size of context encoder")
    parser.add_argument("--decoder_hidden_size", default=100,
        type=int, help="Hidden size of decoder")
    parser.add_argument("--embed_size", default=50,
        type=int, help="Token embedding dimension")
    parser.add_argument("--bidirectional", default=True,
        type=bool, help="Bidirectional model config")
    return parser

def run_memnet_experiment(args):
    vocab = load_memnet_vocabulary(args)

    VEC = SequenceVectorizer(
        vocab,
        init_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]"
        )

    train_dataset = load_memnet_train_data(args, vectorizer)

def run_hred_experiment(args):
    (hred_args, extras) = hred_parser().parse_known_args()
    vocab = load_hred_vocabulary(args)

    # Using the identity function since the input is already tokenized
    VEC = SequenceVectorizer(vocab, 
        tokenizer=lambda x: x,
        init_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]")

    logging.debug("Loading train dataset")
    train_dataset = load_hred_dataset(args, 'train.json', VEC)
    logging.debug("Loading validation dataset")
    val_dataset = load_hred_dataset(args, 'dev.json', VEC)
    logging.debug("Loading test dataset")
    test_dataset = load_hred_dataset(args, 'test.json', VEC)

    datasets = argparse.Namespace(
        train=train_dataset,
        val=val_dataset,
        test=test_dataset)

    loaders = prepare_hred_dataloader(datasets, VEC, args)
    hred_args.device = args.device
    model = create_hred_model(hred_args, vocab).to(args.device)
    logging.info("Model parameters: {}".format(count_parameters(model)))
    optimizer = optim.Adam(model.parameters())
    loss_func = nn.CrossEntropyLoss()
    train_model.train_hred_model(model, optimizer, loss_func, loaders, args)




def main():
    (args, extras) = base_parser().parse_known_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "hred":
        run_hred_experiment(args)

if __name__ == '__main__':
    main()