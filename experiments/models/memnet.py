import torch.nn as nn

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
        pass

    def forward(self, x_in):
        pass

class FactEncoder(nn.Module):
    def __init__(self,
        input_size,
        embed_size):
        pass

    def forward(self, context):
        pass

class InputEncoder(nn.Module):
    def __init__(self, 
        sentence_encoder: SentenceEncoder,
        knowledge_encoder: KnowledgeEncoder):
        self.sentence_encoder = sentence_encoder
        self.knowledge_encoder = knowledge_encoder
        self.softmax = nn.Softmax()

    def forward(self, sentence, context):
        encoded_sentence = self.sentence_encoder(sentence)
        encoded_key, encoded_value = self.knowledge_encoder(context)
        key_probs = self.softmax( encoded_sentence * encoded_key)
        memory_information = key_probs * encoded_value

        return encoded_sentence + memory_information

class Decoder(nn.Module):
    def __init__(self, 
        output_size,
        embed_size,
        hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, 
            batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.output_size = output_size

    def forward(self, y_out, prev_hidden, prev_cell):
        logging.debug("y_out: {}".format(y_out.shape))
        embedded = self.embedding(y_out.unsqueeze(0))
        logging.debug("decoder embedding: {}".format(embedded.shape))
        output, hidden = self.rnn(embedded, (prev_hidden, prev_cell))

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

    def forward(self, source, lengths, target, teacher_forcing_ratio=0.5):
        hidden, cell = self.encoder(source, lengths)
        max_length = source.shape[0]
        batch_size = source.shape[1]
        outputs = torch.zeros(max_length, batch_size, self.decoder.output_size).to(self.device)
        
        decoder_input = target[0, :]
        decoder_hidden = hidden

        for t in range(max_length):
            output, hidden, cell = self.decoder(decoder_input, decoder_hidden, cell)
            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input =  target[t] if teacher_force else output.argmax(1)

        return outputs