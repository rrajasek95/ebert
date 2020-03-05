from vocabulary import Vocabulary
import torch
import logging

def split_tokenizer(x):
    return x.split(" ")

class SequenceVectorizer():
    def __init__(self, vocabulary: Vocabulary, tokenizer=split_tokenizer,
        init_token=None,
        eos_token=None,
        pad_token=None,
        reverse=False):
        self.vocab = vocabulary

        if init_token:
            self.init_idx = vocabulary.add_token(init_token)
            self.init_token = init_token
            self.init_present = 1
        else:
            self.init_present = 0

        if eos_token:
            self.eos_idx = vocabulary.add_token(eos_token)
            self.eos_token = eos_token
            self.eos_present = 1
        else:
            self.eos_present = 0

        if pad_token:
            self.pad_idx = vocabulary.add_token(pad_token)


        self.tokenizer = tokenizer
        self.reverse = reverse


    def vectorize(self, input: str, device = "cpu") -> torch.Tensor :
        """
        Prepare a torch tensor for the input
        """


        tokens = []
        if self.init_token:
            tokens.append(self.init_token)
        tokens += self.tokenizer(input)
        if self.eos_token:
            tokens.append(self.eos_token)
        if self.reverse:
            tokens = list(reversed(tokens))
        logging.debug("Tokens: {}".format(tokens))

        seq_tensor = torch.zeros((len(tokens)), dtype=torch.long)
        
        for i, token in enumerate(tokens):
            idx = self.vocab.lookup_token(token)
            seq_tensor[i] = idx

        logging.debug("Tensor: {}".format(seq_tensor))
        return seq_tensor.to(device)

    def vector_to_sequence(self, input, length):
        sequence = []
        for i in range(1 if self.init_token else 0, int(length) - (1 if self.eos_token else 0)):
            token = input[i]
            word = self.vocab.lookup_index(token.item())
            sequence.append(word)
        return sequence

class OneHotVocabVectorizer(object):
    # Vectorizer that accepts a vocabulary of words and performs one-hot encoding of an utterance

    def __init__(self, vocabulary, dtype=np.float32):
        self.vocabulary = vocabulary
        self.dtype = dtype

    def vectorize(self, utterance):
        one_hot = np.zeros(len(self.vocabulary), dtype=self.dtype)

        for token in utterance.split(" "):
            if token not in string.punctuation:
                one_hot[self.vocabulary.lookup_token(token)] = 1
        return one_hot