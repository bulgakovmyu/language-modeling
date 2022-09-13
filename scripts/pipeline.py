# %%
from src.preprocess.text import TextFileProcessor, make_ngrams
from src.config import Config
from src.models.simple_lang_model import BaseLM
import copy
from src.models.nn_lang_model import NeuralNet, Dataset, LangModel

# %%
tfp = TextFileProcessor(filepath=Config.filepath)
n = 3
# %%
sentences, whole_text_list, vocab = tfp.run()
sentences
# %%
whole_text_list
# %%
#### Custom model
model = BaseLM(n=n)
model.fit(whole_text_list)
# %%
# %%
#### NLTK model to compare
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

lm = MLE(n)
train, vocab = padded_everygram_pipeline(n, sentences)
lm.fit(train, vocab)

# %%
#### NN-model
dataset = Dataset(5)
nn_model = NeuralNet(n_vocab=len(dataset.vocab))

nn_lm = LangModel(dataset, nn_model)
nn_lm.load()

# %%
def generate_all_models(text_seed, length):
    res = model.generate(length, text_seed=text_seed)
    print("custom:::", " ".join(res[0]), ":::", round(res[1]))
    res = model.generate(length, text_seed=text_seed, smoothing=0.01)
    print("custom_sm_001:::", " ".join(res[0]), ":::", round(res[1]))
    res = model.generate(length, text_seed=text_seed, smoothing=0.5)
    print("custom_sm_05:::", " ".join(res[0]), ":::", round(res[1]))
    res = model.generate(length, text_seed=text_seed, smoothing=1)
    print("custom_sm_1:::", " ".join(res[0]), ":::", round(res[1]))
    res = copy.deepcopy(text_seed)
    res.extend(lm.generate(length - len(text_seed), text_seed=text_seed))
    print(
        "nltk:::",
        " ".join(res),
        ":::",
        round(lm.perplexity(make_ngrams(res, n)))
        if lm.perplexity(make_ngrams(res, n)) < 10e50
        else "inf",
    )
    res = nn_lm.predict(text_seed, next_words=length - len(text_seed))
    print("nn:::", " ".join(res))
    print("------=------=------=------=------=------=------=------=------")


# %%
init_seed = ["i", "am"]
for i in range(3):
    generate_all_models(init_seed, 7)
# %%
test = ["let", "us", "kill"]
for i in range(3):
    generate_all_models(test, 5)
# %%
test = ["let"]
for i in range(3):
    generate_all_models(test, 4)
# %%
test = ["i", "would"]
for i in range(3):
    generate_all_models(test, 4)

# %%
test = ["he", "should"]
for i in range(3):
    generate_all_models(test, 4)
# %%
test = ["i", "have"]
for i in range(3):
    generate_all_models(test, 4)
# %%
