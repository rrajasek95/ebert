import torch
import torch.nn as nn
import logging
import random
"""
memnet.py

This file implements the model from 
"A Knowledge-Grounded Neural Conversation Model" (Ghazvininejad et al. 2018)
which is a Seq2Seq based memory network
https://arxiv.org/pdf/1702.01932.pdf

The model comprises of:
1. A sentence encoder
2. Memory parameters that encode a representation for selecting relevant facts
3. Decoder that takes an encoded state and generates an utterance
"""


class SentenceEncoder(nn.Module):
    def __init__(self,
        input_size,
        embed_size,
        hidden_size,
        bidirectional,
        dropout):
        super(SentenceEncoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(
            input_size=embed_size, 
            hidden_size=hidden_size, 
            batch_first=True, bidirectional=bidirectional)
        self.hidden_size = hidden_size

    def forward(self, x_in):
        # Currently we have [seq_len]
        logging.debug("Shape: {}".format(x_in.shape))
        # It needs to become [batch_size, seq_len], so we add an extra dimension
        x = x_in.unsqueeze(0)
        # x: [1, seq_len]
        embedded = self.embedding(x)
        # embedded: [1, seq_len, embed_size]
        logging.debug("embedded: {}".format(embedded.shape))
        output, (hidden, cell) = self.rnn(embedded)
        logging.debug("hidden: {}".format(hidden.shape))

        return hidden

class FactEncoder(nn.Module):
    def __init__(self,
        input_size,
        embed_size):
        super(FactEncoder, self).__init__()
        self.memory = nn.Linear(input_size, embed_size, bias=False)
        self.value = nn.Linear(input_size, embed_size, bias=False)

    def forward(self, context):
        # [num_facts, vocab_size]
        logging.debug("Context shape: {}".format(context.shape))
        context = context.unsqueeze(0)
        # [1, num_facts, vocab_size]
        memory = self.memory(context)
        # [1, num_facts, embed_size]

        value = self.value(context)
        # [1, num_facts, embed_size]
        return memory, value

class InputEncoder(nn.Module):
    def __init__(self, 
        sentence_encoder: SentenceEncoder,
        fact_encoder: FactEncoder):
        super(InputEncoder, self).__init__()
        self.sentence_encoder = sentence_encoder
        self.fact_encoder = fact_encoder
        self.softmax = nn.Softmax()

    def forward(self, sentence, context):
        logging.debug("Inputencoder: {}".format(sentence.shape))
        logging.debug("Inputencoder context: {}".format(context.shape))
        encoded_sentence = self.sentence_encoder(sentence).view(-1)
        encoded_key, encoded_value = self.fact_encoder(context)
        
        encoded_key = encoded_key.squeeze(0)
        encoded_value = encoded_value.squeeze(0)
        logging.debug(encoded_sentence.shape)
        logging.debug(encoded_key.shape)
        logging.debug(encoded_value.shape)

        key_product = torch.mv(encoded_key, encoded_sentence)
        logging.debug(key_product.shape)
        key_probs = self.softmax( key_product)

        logging.debug("Key probs shape: {}".format(key_probs.shape))
        logging.debug("Key probs: {}".format(key_probs))
        key_probs = key_probs.unsqueeze(0)
        memory_information = torch.mm(key_probs, encoded_value).squeeze(0)
        logging.debug("Memory info shape: {}".format(memory_information.shape))
        return encoded_sentence + memory_information

class Decoder(nn.Module):
    def __init__(self, 
        output_size,
        embed_size,
        hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, 
            batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.output_size = output_size

    def forward(self, y_out, prev_hidden):
        logging.debug("y_out: {}".format(y_out.shape))
        logging.debug("device: {}".format(y_out.device))
        embedded = self.embedding(y_out.unsqueeze(0))
        logging.debug("decoder embedding: {}".format(embedded.shape))
        output, hidden = self.rnn(embedded, prev_hidden)

        pred = self.fc_out(output.squeeze(0))
        return pred, hidden


class LSTMSeq2Seq(nn.Module):
    """
    Special seq2seq model to handle LSTM models which have a cell state
    """
    def __init__(self, encoder, decoder, device):
        super(LSTMSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, context, target, teacher_forcing_ratio=0.5):
        outputs = torch.zeros(len(source), target[0].shape[0], self.decoder.output_size, device=self.device)
        x_in = zip(source, context)
        for i, (s, c) in enumerate(x_in):
            encoded_sentence = self.encoder(s, c)
            max_length = target[0].shape[0]

            logging.debug("Target {}".format(target[i]))
            logging.debug("Target {}".format(target[i][0].shape))
            decoder_input = target[i][0].unsqueeze(0)
            decoder_hidden = encoded_sentence.unsqueeze(0).unsqueeze(0)

            for t in range(max_length):
                logging.debug("Decoder input: {}".format(decoder_input.shape))
                output, hidden = self.decoder(decoder_input, decoder_hidden)
                logging.debug("Outputs shape: {}".format(outputs.shape))
                logging.debug("Output shape: {}".format(output.shape))
                logging.debug("Hidden: {}".format(hidden.shape))
                outputs[i][t] = output
                # teacher_force = random.random() < teacher_forcing_ratio
                decoder_input = output.argmax(1)
                logging.debug("Decoder input shape: {}".format(decoder_input.shape))
        return outputs