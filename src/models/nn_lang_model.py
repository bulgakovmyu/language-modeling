import copy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import numpy as np
import pandas as pd
from src.preprocess.text import TextFileProcessor
from src.config import Config


class NeuralNet(nn.Module):
    def __init__(self, n_vocab):
        super(NeuralNet, self).__init__()
        self.lstm_size = 128
        self.embedding_size = 128
        self.num_layers = 3

        self.embedding = nn.Embedding(
            num_embeddings=n_vocab, embedding_dim=self.embedding_size
        )

        self.lstm_layer = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc_layer = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        emb = self.embedding(x)
        output, state = self.lstm_layer(emb, prev_state)
        logits = self.fc_layer(output)

        return logits, state

    def initialize_state(self, sequence_length):
        return (
            torch.zeros(self.num_layers, sequence_length, self.lstm_size),
            torch.zeros(self.num_layers, sequence_length, self.lstm_size),
        )


class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.load_text()

        self.index_to_word = {index: word for index, word in enumerate(self.vocab)}
        self.word_to_index = {word: index for index, word in enumerate(self.vocab)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_text(self):
        tfp = TextFileProcessor(filepath=Config.filepath)
        _, self.words, self.vocab = tfp.run()

    def __len__(self):
        return len(self.words_indexes) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index : index + self.sequence_length]),
            torch.tensor(
                self.words_indexes[index + 1 : index + self.sequence_length + 1]
            ),
        )


class LangModel(object):
    def __init__(self, dataset, model, batch_size=512, sequence_length=5, max_epoch=5):
        self.sequence_length = sequence_length
        self.max_epoch = max_epoch
        self.dataloader = DataLoader(dataset, batch_size=batch_size)
        self.word_to_index = dataset.word_to_index
        self.index_to_word = dataset.index_to_word
        self.model = model
        self.init_train_params()

    def init_train_params(self):
        self.model.train()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    def fit(self):
        res = []
        print(f"{len(self.dataloader)} batches for each epoch")
        for epoch in range(self.max_epoch):
            print("LR: ", self.optimizer.param_groups[0]["lr"])
            state_h, state_c = self.model.initialize_state(self.sequence_length)
            losses = []
            for batch, (x, y) in enumerate(self.dataloader):
                self.optimizer.zero_grad()

                y_pred, (state_h, state_c) = self.model(x, (state_h, state_c))
                loss = self.criterion(y_pred.transpose(1, 2), y)

                state_h = state_h.detach()
                state_c = state_c.detach()

                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                if batch % 50 == 0:
                    res.append(
                        {"epoch": epoch, "batch": batch, "loss": np.mean(losses)}
                    )
                    print({"epoch": epoch, "batch": batch, "loss": np.mean(losses)})
                    losses = []
            self.scheduler.step()

        pd.DataFrame(res)[["loss"]].plot(kind="line")
        plt.show()
        torch.save(self.model.state_dict(), Config.modelpath)
        return res

    def load(self):
        self.model.load_state_dict(torch.load(Config.modelpath))

    def predict(self, text_array, next_words):
        self.model.eval()

        state_h, state_c = self.model.initialize_state(len(text_array))
        text_array = copy.deepcopy(text_array)
        for i in range(0, next_words):
            x = torch.tensor([[self.word_to_index[w] for w in text_array[i:]]])
            y_pred, (state_h, state_c) = self.model(x, (state_h, state_c))

            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            text_array.append(self.index_to_word[word_index])

        return text_array
