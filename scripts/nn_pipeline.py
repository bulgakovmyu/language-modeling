# %%
from src.models.nn_lang_model import NeuralNet, Dataset, LangModel
import torch

# %%
sequence_length = 5
batch_size = 512
max_epoch = 5

dataset = Dataset(sequence_length)
model = NeuralNet(n_vocab=len(dataset.vocab))

lm = LangModel(dataset, model, batch_size, sequence_length, max_epoch)

# %%
res = lm.fit()

# %%
def generate_all_models(text_seed, length):
    res = lm.predict(text_seed, next_words=length - len(text_seed))
    print("nn:::", " ".join(res))


# %%
for i in range(0, 5):
    generate_all_models(["let", "us", "kill"], length=5)
# %%
for i in range(0, 5):
    generate_all_models(["i", "am"], length=5)

# %%
for i in range(0, 5):
    generate_all_models(["i", "would"], length=5)
# %%
for i in range(0, 5):
    generate_all_models(["he", "should"], length=5)

# %%
for i in range(0, 5):
    generate_all_models(["i", "have"], length=5)
# %%
