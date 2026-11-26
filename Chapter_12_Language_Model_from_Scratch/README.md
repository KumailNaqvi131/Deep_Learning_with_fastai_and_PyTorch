# Chapter 12: A Language Model from Scratch

Building an RNN language model step by step on a tiny dataset, the numbers from one to 9,999 written in English, up to a regularized, weight-tied LSTM (the ideas behind AWD-LSTM).

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | Human Numbers: one stream of tokens, a vocab of 30 words | The Data | `01_the_data.py` |
| 2 | 3 words -> next word, with shared weights (`LMModel1`); the most-common-token baseline | Our First Language Model from Scratch, Our Language Model in PyTorch | `02_first_language_model.py` |
| 3 | The same model written as a loop: an RNN (`LMModel2`) | Our First Recurrent Neural Network | `03_first_rnn.py` |
| 4 | Keeping the hidden state between batches, `detach` (truncated BPTT), `group_chunks`, `ModelResetter` | Maintaining the State of an RNN | `04_maintaining_state.py` |
| 5 | Predicting after every word (`LMModel4`) | Creating More Signal | `05_creating_more_signal.py` |
| 6 | `nn.RNN` with 2 layers, and why deep RNNs explode or vanish | Multilayer RNNs, Exploding or Disappearing Activations | `06_multilayer_rnns.py` |
| 7 | An LSTM cell from scratch (checked against `nn.LSTMCell`), a 2-layer LSTM | LSTM, Building an LSTM from Scratch, Training a Language Model Using LSTMs | `07_lstm.py` |
| 8 | Dropout from scratch, AR and TAR, weight tying, weight decay | Regularizing an LSTM ... Training a Weight-Tied Regularized LSTM | `08_regularizing_an_lstm.py` |

`human_numbers.py` holds the shared data and DataLoaders.

## Run

```bash
cd Chapter_12_Language_Model_from_Scratch
python 01_the_data.py
python 02_first_language_model.py
python 03_first_rnn.py
python 04_maintaining_state.py
python 05_creating_more_signal.py
python 06_multilayer_rnns.py
python 07_lstm.py
python 08_regularizing_an_lstm.py
```

The dataset is generated exactly like the real one, and results follow the book's progression: about 49% (3 words), 60% (stateful, more signal), 54% (2-layer RNN: worse!), 79% (LSTM), 84% (regularized LSTM). Each script takes seconds, also on a CPU.

## Key takeaways
- An RNN is a loop that reuses the same layers at every step and carries a hidden state.
- Keep the state between batches, but detach it so gradients only flow back a fixed number of steps.
- Deep RNNs multiply by the same matrices many times, so activations and gradients explode or vanish. LSTMs fix this with gated cell states.
- Dropout, activation regularization (AR/TAR) and weight tying make LSTMs generalise much better.
