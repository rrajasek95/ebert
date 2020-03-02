from torch.utils.data import Dataset
import logging

class MixedShortDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        row = self.data[index]["paragraphs"][0]
        context = row["context"]
        qa = row["qas"][0]
        utterance = qa["question"]
        response = qa["answers"][0]["text"]

class HredDataset(Dataset):
    def __init__(self, contexts, responses,
        vectorizer, device):
        self.contexts = contexts
        self.responses = responses
        self.vectorizer = vectorizer
        self.device = device

    def __getitem__(self, index):
        context = self.contexts[index]
        response = self.responses[index]

        x_data = [self.vectorizer.vectorize(c, self.device) for c in context]
        x_lengths = [len(c) for c in context]
        y_target = self.vectorizer.vectorize(response, self.device)
        return {
            "x_data": x_data,
            "y_target": y_target
        }

    def __len__(self):
        return len(self.contexts)

    def get_num_batches(self, batch_size):
        return len(self) // batch_size