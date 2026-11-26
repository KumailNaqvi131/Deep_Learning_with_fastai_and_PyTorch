"""NLP with fastai's approach (Ch. 10, 12, 15), in plain PyTorch:

  tokenize (with fastai's special tokens xxbos, xxmaj, xxup, xxrep, ...) -> numericalize (Vocab) ->
  language-model batches (one long stream of text) or classifier batches (padded, sorted by length) ->
  AWD-LSTM language model -> fine-tune -> save the encoder -> classifier (ULMFiT).
"""

import html
import math
import random
import re
import warnings
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .learner import Callback, CrossEntropyLossFlat, DataLoaders, DeviceDataLoader, Learner
from .utils import DATA_ROOT, get_device

# ==================================================================================================
# Tokenization
# ==================================================================================================
UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ = "xxunk xxpad xxbos xxeos xxfld xxrep xxwrep xxup xxmaj".split()
defaults_text_spec_tok = [UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ]


def fix_html(x):
    """Replace HTML leftovers such as <br /> and &amp; (IMDb reviews contain them)."""
    x = x.replace("#39;", "'").replace("amp;", "&").replace("#146;", "'").replace("nbsp;", " ")
    x = x.replace("#36;", "$").replace("\\n", "\n").replace("quot;", "'").replace("<br />", "\n")
    x = x.replace('\\"', '"').replace("<unk>", UNK).replace(" @.@ ", ".").replace(" @-@ ", "-")
    return html.unescape(x)


def replace_rep(t):
    """'!!!!!' -> ' xxrep 5 ! ' : a character repeated 4+ times."""
    return re.sub(r"(\S)(\1{2,})", lambda m: f" {TK_REP} {len(m.group(2)) + 1} {m.group(1)} ", t)


def replace_wrep(t):
    """'very very very' -> ' xxwrep 3 very ' : a word repeated 3+ times."""
    return re.sub(r"(?:\s|^)(\w+)\s+((?:\1\s+)+)\1(\s|\W|$)",
                  lambda m: f" {TK_WREP} {len(m.group(2).split()) + 2} {m.group(1)} {m.group(3)}", t)


def spec_add_spaces(t):
    return re.sub(r"([/#\\])", r" \1 ", t)


def rm_useless_spaces(t):
    return re.sub(" {2,}", " ", t)


def lowercase(t, add_bos=True):
    return (f"{BOS} " if add_bos else "") + t.strip().lower()


defaults_text_proc_rules = [fix_html, replace_rep, replace_wrep, spec_add_spaces, rm_useless_spaces]

_TOKEN_RE = re.compile(r"xx\w+|n't|'s|'re|'ve|'ll|'d|'m|\w+(?=n't)|\w+|[^\w\s]|\n")


def word_tokenize(t):
    """A simple rule-based word tokenizer (the book uses spaCy; this behaves similarly on English):
    splits punctuation, and contractions like don't -> do n't, it's -> it 's."""
    return _TOKEN_RE.findall(t)


def replace_all_caps(toks):
    """'GREAT' -> 'xxup great'."""
    res = []
    for t in toks:
        if t.isupper() and len(t) > 1 and t.isalpha():
            res += [TK_UP, t.lower()]
        else:
            res.append(t)
    return res


def replace_maj(toks):
    """'Great' -> 'xxmaj great' (the capital letter becomes a token)."""
    res = []
    for t in toks:
        if len(t) > 0 and t[0].isupper() and t[1:].islower():
            res += [TK_MAJ, t.lower()]
        else:
            res.append(t.lower() if not t.startswith("xx") else t)
    return res


defaults_text_postproc_rules = [replace_all_caps, replace_maj]


class Tokenizer:
    """Pre-rules on the string -> word tokenizer -> post-rules on the tokens, plus xxbos at the start."""

    def __init__(self, tok=word_tokenize, rules=None, post_rules=None, add_bos=True):
        self.tok, self.add_bos = tok, add_bos
        self.rules = defaults_text_proc_rules if rules is None else rules
        self.post_rules = defaults_text_postproc_rules if post_rules is None else post_rules

    def __call__(self, text):
        for r in self.rules:
            text = r(text)
        toks = self.tok(text)
        for r in self.post_rules:
            toks = r(toks)
        return ([BOS] if self.add_bos else []) + [t for t in toks if t.strip() or t == "\n"]


def detokenize(toks):
    """Turn tokens back into readable text (undo xxmaj/xxup, drop xxbos/xxpad)."""
    out, cap, up = [], False, False
    toks, expanded, i = list(toks), [], 0
    while i < len(toks):                                # xxrep 3 ! -> !!!   xxwrep 3 very -> very very very
        if toks[i] in (TK_REP, TK_WREP) and i + 2 < len(toks) and toks[i + 1].isdigit():
            n, w = int(toks[i + 1]), toks[i + 2]
            expanded += [w * n] if toks[i] == TK_REP else [w] * n
            i += 3
        else:
            expanded.append(toks[i])
            i += 1
    for t in expanded:
        if t in (BOS, PAD, EOS):
            continue
        if t == TK_MAJ:
            cap = True
            continue
        if t == TK_UP:
            up = True
            continue
        w = t.upper() if up else (t.capitalize() if cap else t)
        cap = up = False
        if out and (re.match(r"^[^\w\s(]", w) or w in ("n't", "'s", "'re", "'ve", "'ll", "'d", "'m")):
            out[-1] += w
        else:
            out.append(w)
    return " ".join(out).replace(" \n ", "\n")


# ==================================================================================================
# Numericalization
# ==================================================================================================
class Numericalize:
    """Token -> integer index. The vocab holds the special tokens first, then every token seen at least
    `min_freq` times (most frequent first), at most `max_vocab` tokens. Unknown tokens -> xxunk (0)."""

    def __init__(self, vocab=None, min_freq=3, max_vocab=60000):
        self.vocab, self.min_freq, self.max_vocab = vocab, min_freq, max_vocab

    def setup(self, token_lists):
        if self.vocab is None:
            count = Counter(t for toks in token_lists for t in toks)
            words = [w for w, c in count.most_common(self.max_vocab) if c >= self.min_freq and w not in defaults_text_spec_tok]
            self.vocab = defaults_text_spec_tok + words
            while len(self.vocab) % 8:                      # a multiple of 8 is faster on GPUs
                self.vocab.append(f"xxfake{len(self.vocab)}")
        self.o2i = {w: i for i, w in enumerate(self.vocab)}
        return self

    def __call__(self, toks):
        return torch.tensor([self.o2i.get(t, 0) for t in toks], dtype=torch.long)

    def decode(self, idxs):
        return [self.vocab[int(i)] for i in idxs]


# ==================================================================================================
# Batches
# ==================================================================================================
class LMDataLoader:
    """Language-model batches (Ch. 10 and 12): concatenate all the texts (shuffled on the training set)
    into one stream, cut it into `bs` equal rows, and walk along the rows `seq_len` tokens at a time.
    The target is the input shifted by one token. Row i of batch k+1 continues row i of batch k."""

    def __init__(self, docs, bs=64, seq_len=72, shuffle=False):
        self.docs, self.bs, self.seq_len, self.shuffle = docs, bs, seq_len, shuffle
        self.n_tok = sum(len(d) for d in docs)
        self.dataset = docs

    def __len__(self):
        return math.ceil(((self.n_tok - 1) // self.bs) / self.seq_len)

    def __iter__(self):
        docs = list(self.docs)
        if self.shuffle:
            random.shuffle(docs)
        stream = torch.cat(docs)
        per_row = (len(stream) - 1) // self.bs
        x = stream[:per_row * self.bs].view(self.bs, per_row)
        y = stream[1:per_row * self.bs + 1].view(self.bs, per_row)
        for i in range(0, per_row, self.seq_len):
            yield x[:, i:i + self.seq_len], y[:, i:i + self.seq_len]


def pad_collate(samples, pad_idx=1, pad_first=True):
    """Pad token sequences to the longest one in the batch (padding at the start, like fastai)."""
    max_len = max(len(s[0]) for s in samples)
    x = torch.full((len(samples), max_len), pad_idx, dtype=torch.long)
    for i, s in enumerate(samples):
        if pad_first:
            x[i, max_len - len(s[0]):] = s[0]
        else:
            x[i, :len(s[0])] = s[0]
    if len(samples[0]) == 1:
        return (x,)
    return x, torch.stack([torch.as_tensor(s[1]) for s in samples])


class SortedDataLoader:
    """Batches of texts of similar length (less padding). On the training set the order is still random:
    shuffle, take chunks of 50 batches, sort each chunk by length, then shuffle the batches."""

    def __init__(self, items, bs=64, shuffle=False, pad_idx=1):
        self.items, self.bs, self.shuffle, self.pad_idx = items, bs, shuffle, pad_idx
        self.dataset = items

    def __len__(self):
        return math.ceil(len(self.items) / self.bs)

    def __iter__(self):
        idx = list(range(len(self.items)))
        key = lambda i: -len(self.items[i][0])  # noqa: E731
        if self.shuffle:
            random.shuffle(idx)
            chunks = [sorted(idx[i:i + self.bs * 50], key=key) for i in range(0, len(idx), self.bs * 50)]
            batches = [c[j:j + self.bs] for c in chunks for j in range(0, len(c), self.bs)]
            random.shuffle(batches)
        else:
            idx = sorted(idx, key=key)
            batches = [idx[j:j + self.bs] for j in range(0, len(idx), self.bs)]
        for b in batches:
            yield pad_collate([self.items[i] for i in b], self.pad_idx)


class TextDataLoaders(DataLoaders):
    def show_batch(self, max_n=4, trunc=60):
        b = self.train.one_batch()
        for i in range(min(max_n, len(b[0]))):
            text = detokenize(self.num.decode(b[0][i]))
            if self.is_lm:
                nxt = " ".join(self.num.decode(b[1][i][-5:]))
                print(f"text: ...{text[-trunc * 2:]!r}\n   -> last 5 targets: {nxt}")
            else:
                print(f"{self.vocab[int(b[1][i])]:>6} | {text[-trunc * 3:]!r}")


def get_text_files(path, folders=None):
    path = Path(path)
    roots = [path / f for f in folders] if folders else [path]
    return sorted(p for r in roots for p in r.rglob("*.txt"))


def lm_dataloaders(texts, valid_pct=0.1, bs=64, seq_len=72, vocab=None, seed=42, min_freq=3, tokenizer=None):
    """DataLoaders for a language model: every text is used, no labels needed (Ch. 10)."""
    tok = tokenizer or Tokenizer()
    toks = [tok(t) for t in texts]
    num = Numericalize(vocab, min_freq=min_freq).setup(toks)
    docs = [num(t) for t in toks]
    g = random.Random(seed)
    idx = list(range(len(docs)))
    g.shuffle(idx)
    cut = int(len(idx) * valid_pct)
    tr, va = [docs[i] for i in idx[cut:]], [docs[i] for i in idx[:cut]]
    device = get_device()
    dls = TextDataLoaders(DeviceDataLoader(LMDataLoader(tr, bs, seq_len, shuffle=True), device, train=True),
                          DeviceDataLoader(LMDataLoader(va, bs, seq_len), device), vocab=num.vocab,
                          loss_func=CrossEntropyLossFlat(), device=device)
    dls.num, dls.tok, dls.is_lm, dls.seq_len = num, tok, True, seq_len
    return dls


def clas_dataloaders(train_texts, train_labels, valid_texts, valid_labels, text_vocab=None, bs=64, min_freq=3,
                     tokenizer=None):
    """DataLoaders for a text classifier. Pass the language model's vocab so the token ids match (Ch. 10)."""
    tok = tokenizer or Tokenizer()
    tr_toks, va_toks = [tok(t) for t in train_texts], [tok(t) for t in valid_texts]
    num = Numericalize(text_vocab, min_freq=min_freq).setup(tr_toks)
    labels = sorted(set(train_labels))
    l2i = {lbl: i for i, lbl in enumerate(labels)}
    tr = [(num(t), l2i[y]) for t, y in zip(tr_toks, train_labels)]
    va = [(num(t), l2i[y]) for t, y in zip(va_toks, valid_labels)]
    device = get_device()
    dls = TextDataLoaders(DeviceDataLoader(SortedDataLoader(tr, bs, shuffle=True), device, train=True),
                          DeviceDataLoader(SortedDataLoader(va, bs * 2), device), vocab=labels,
                          loss_func=CrossEntropyLossFlat(), device=device)
    dls.num, dls.tok, dls.is_lm, dls.text_vocab = num, tok, False, num.vocab
    return dls


def imdb_texts(path, split, labels=("neg", "pos")):
    files = [f for lbl in labels for f in sorted((Path(path) / split / lbl).glob("*.txt"))]
    return [f.read_text(encoding="utf-8") for f in files], [f.parent.name for f in files]


# ==================================================================================================
# AWD-LSTM (Merity et al.): an LSTM with lots of dropout, the model behind ULMFiT
# ==================================================================================================
class RNNDropout(nn.Module):
    """Dropout with the same mask for every time step (x: [bs, seq_len, n])."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p) / (1 - self.p)
        return x * mask


class WeightDropout(nn.Module):
    """DropConnect on the LSTM's hidden-to-hidden weights: drop *weights* rather than activations."""

    def __init__(self, module, weight_p, layer_names=("weight_hh_l0",)):
        super().__init__()
        self.module, self.weight_p, self.layer_names = module, weight_p, layer_names
        for name in layer_names:
            w = getattr(self.module, name)
            delattr(self.module, name)
            self.register_parameter(f"{name}_raw", nn.Parameter(w.data))
            setattr(self.module, name, w.data.clone())

    def _setweights(self):
        for name in self.layer_names:
            raw = getattr(self, f"{name}_raw")
            # (clone in eval mode: assigning the Parameter itself would re-register it inside the LSTM)
            w = F.dropout(raw, p=self.weight_p) if self.training else raw.clone()
            setattr(self.module, name, w)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)   # "weights not contiguous" from cuDNN
            return self.module(*args)


class EmbeddingDropout(nn.Module):
    """Drop entire rows of the embedding matrix (whole words) during training."""

    def __init__(self, emb, embed_p):
        super().__init__()
        self.emb, self.embed_p = emb, embed_p

    def forward(self, words):
        w = self.emb.weight
        if self.training and self.embed_p != 0:
            mask = w.new_empty(w.size(0), 1).bernoulli_(1 - self.embed_p) / (1 - self.embed_p)
            w = w * mask
        return F.embedding(words, w, self.emb.padding_idx)


awd_lstm_lm_config = dict(emb_sz=400, n_hid=1152, n_layers=3, pad_token=1, bidir=False, output_p=0.1,
                          hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True)
awd_lstm_clas_config = dict(emb_sz=400, n_hid=1152, n_layers=3, pad_token=1, bidir=False, output_p=0.4,
                            hidden_p=0.3, input_p=0.4, embed_p=0.05, weight_p=0.5)


class AWD_LSTM(nn.Module):
    def __init__(self, vocab_sz, emb_sz=400, n_hid=1152, n_layers=3, pad_token=1, hidden_p=0.2, input_p=0.6,
                 embed_p=0.1, weight_p=0.5, **kwargs):
        super().__init__()
        self.bs, self.emb_sz, self.n_hid, self.n_layers = 1, emb_sz, n_hid, n_layers
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder.weight.data.uniform_(-0.1, 0.1)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        self.rnns = nn.ModuleList([WeightDropout(nn.LSTM(emb_sz if l == 0 else n_hid,
                                                         emb_sz if l == n_layers - 1 else n_hid, 1, batch_first=True),
                                                 weight_p) for l in range(n_layers)])
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for _ in range(n_layers)])
        self.reset()

    def forward(self, inp):
        bs, sl = inp.shape
        if bs != self.bs or self.hidden is None:
            self.bs = bs
            self.reset()
        out = self.input_dp(self.encoder_dp(inp))
        new_hidden = []
        self.raw_outputs, self.outputs = [], []
        for l, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            out, h = rnn(out, self.hidden[l] if self.hidden is not None else None)
            new_hidden.append(tuple(t.detach() for t in h))     # truncated backprop through time
            self.raw_outputs.append(out)
            if l != self.n_layers - 1:
                out = hid_dp(out)
            self.outputs.append(out)
        self.hidden = new_hidden
        return out

    def reset(self):
        """Forget the hidden state (start of an epoch / a new text)."""
        self.hidden = None


class LinearDecoder(nn.Module):
    """Hidden state -> scores for every word of the vocab. Weights tied to the input embedding."""

    def __init__(self, n_out, n_hid, output_p=0.1, tie_encoder=None, bias=True):
        super().__init__()
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        self.decoder.weight.data.uniform_(-0.1, 0.1)
        self.output_dp = RNNDropout(output_p)
        if bias:
            self.decoder.bias.data.zero_()
        if tie_encoder is not None:
            self.decoder.weight = tie_encoder.weight

    def forward(self, x):
        return self.decoder(self.output_dp(x))


class SequentialRNN(nn.Sequential):
    def reset(self):
        for c in self.children():
            if hasattr(c, "reset"):
                c.reset()


class PoolingLinearClassifier(nn.Module):
    """Classifier head: concat [last hidden state, max pool, average pool] over the non-padding
    time steps, then BatchNorm/Dropout/Linear layers (Ch. 10 and 15)."""

    def __init__(self, dims, ps, pad_idx=1):
        super().__init__()
        layers = []
        for i, (ni, no, p) in enumerate(zip(dims[:-1], dims[1:], ps)):
            layers += [nn.BatchNorm1d(ni), nn.Dropout(p), nn.Linear(ni, no)]
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.layers, self.pad_idx = nn.Sequential(*layers), pad_idx

    def forward(self, inp_out):
        inp, out = inp_out
        mask = (inp == self.pad_idx).unsqueeze(-1)                     # True for padding
        n = (~mask).sum(1).clamp(min=1)
        avg = out.masked_fill(mask, 0).sum(1) / n
        mx = out.masked_fill(mask, -float("inf")).max(1)[0]
        x = torch.cat([out[:, -1], mx, avg], 1)                         # padding is at the start
        return self.layers(x)


class SentenceEncoder(nn.Module):
    """Run the AWD-LSTM over the whole text (resetting the state for each batch of texts) and return
    (input ids, outputs) for the pooling head. fastai processes long texts in chunks of `bptt` tokens;
    the reviews here are short enough to process in one go (Chapter 15 shows the chunked version)."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inp):
        self.module.reset()
        return inp, self.module(inp)

    def reset(self):
        self.module.reset()


def get_language_model(vocab_sz, config=None, drop_mult=1.0):
    cfg = {**awd_lstm_lm_config, **(config or {})}
    for k in cfg:
        if k.endswith("_p"):
            cfg[k] *= drop_mult
    enc = AWD_LSTM(vocab_sz, **cfg)
    dec = LinearDecoder(vocab_sz, cfg["emb_sz"], cfg["output_p"], tie_encoder=enc.encoder if cfg["tie_weights"] else None)
    return SequentialRNN(enc, dec)


def get_text_classifier(vocab_sz, n_class, config=None, drop_mult=1.0, lin_ftrs=(50,), ps=(0.1,)):
    cfg = {**awd_lstm_clas_config, **(config or {})}
    for k in cfg:
        if k.endswith("_p"):
            cfg[k] *= drop_mult
    enc = AWD_LSTM(vocab_sz, **cfg)
    dims = [cfg["emb_sz"] * 3] + list(lin_ftrs) + [n_class]
    return SequentialRNN(SentenceEncoder(enc), PoolingLinearClassifier(dims, [cfg["output_p"]] + list(ps)))


def awd_lstm_lm_split(model):
    """Parameter groups for gradual unfreezing: one per LSTM layer, then embedding + decoder (tied)."""
    enc, dec = model[0], model[1]
    groups = [list(rnn.parameters()) for rnn in enc.rnns]
    return groups + [list(enc.encoder.parameters()) + [p for n, p in dec.named_parameters() if "weight" not in n]]


def awd_lstm_clas_split(model):
    enc = model[0].module
    return [list(enc.encoder.parameters())] + [list(r.parameters()) for r in enc.rnns] + [list(model[1].parameters())]


class ModelResetter(Callback):
    """Reset the RNN's hidden state at the start of training, validation and after fitting."""

    def before_train(self):
        self.learn.model.reset()

    def before_validate(self):
        self.learn.model.reset()

    def after_fit(self):
        self.learn.model.reset()


class RNNRegularizer(Callback):
    """Activation regularization (AR) and temporal activation regularization (TAR), Ch. 12."""

    def __init__(self, alpha=2.0, beta=1.0):
        self.alpha, self.beta = alpha, beta

    def after_loss(self):
        if not self.learn.training:
            return
        enc = self.learn.model[0]
        enc = getattr(enc, "module", enc)
        if self.alpha:
            self.learn.loss = self.learn.loss + self.alpha * enc.outputs[-1].float().pow(2).mean()
        if self.beta:
            h = enc.raw_outputs[-1].float()
            if h.size(1) > 1:
                self.learn.loss = self.learn.loss + self.beta * (h[:, 1:] - h[:, :-1]).pow(2).mean()


def perplexity(inp, targ):
    """exp(cross-entropy): 'how many words is the model hesitating between?'"""
    return torch.exp(F.cross_entropy(inp.reshape(-1, inp.shape[-1]).float(), targ.reshape(-1)))


class TextLearner(Learner):
    def save_encoder(self, name):
        enc = self.model[0]
        enc = getattr(enc, "module", enc)
        fname = self.path / f"{name}.pth"
        fname.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"encoder": enc.state_dict(), "vocab": self.dls.num.vocab}, fname)
        return fname

    def load_encoder(self, name):
        state = torch.load(self.path / f"{name}.pth", map_location=self.dls.device)
        enc = self.model[0]
        enc = getattr(enc, "module", enc)
        enc.load_state_dict(match_embeds(state["encoder"], state["vocab"], self.dls.num.vocab))
        return self

    def predict(self, text, n_words=None, temperature=1.0):
        """Classifier: (label, index, probabilities). Language model: the text continued by n_words."""
        self.model.eval()
        if n_words is not None:
            return self.generate(text, n_words, temperature)
        x = self.dls.num(self.dls.tok(text)).unsqueeze(0).to(self.dls.device)
        with torch.no_grad():
            probs = F.softmax(self.model(x).float(), dim=-1)[0].cpu()
        i = int(probs.argmax())
        return self.dls.vocab[i], torch.tensor(i), probs

    def generate(self, text, n_words=40, temperature=0.75):
        toks = self.dls.tok(text)
        idxs = self.dls.num(toks).to(self.dls.device)
        self.model.reset()
        out = []
        with torch.no_grad():
            logits = self.model(idxs.unsqueeze(0))[0, -1]
            for _ in range(n_words):
                logits[0] = -float("inf")                            # never generate xxunk
                probs = F.softmax(logits.float() / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
                out.append(int(nxt))
                logits = self.model(nxt.view(1, 1))[0, -1]
        self.model.reset()
        return detokenize(toks + self.dls.num.decode(out))


def match_embeds(state, old_vocab, new_vocab):
    """Re-order the pretrained embedding rows to match a new vocab. Words the pretrained model never saw
    get the mean embedding (this is how fastai reuses WikiText-103 weights on IMDb)."""
    old = state["encoder.weight"]
    o2i = {w: i for i, w in enumerate(old_vocab)}
    new = old.new_zeros(len(new_vocab), old.size(1))
    mean = old.mean(0)
    for i, w in enumerate(new_vocab):
        new[i] = old[o2i[w]] if w in o2i else mean
    state = dict(state)
    for key in ("encoder.weight", "encoder_dp.emb.weight"):     # the same matrix, registered twice
        if key in state:
            state[key] = new
    return state


def language_model_learner(dls, arch=AWD_LSTM, config=None, drop_mult=1.0, pretrained=True, pretrained_fname=None,
                           metrics=None, **kwargs):
    """An AWD-LSTM language model. pretrained=True loads `pretrained_fname` (an encoder saved with
    save_encoder on a general corpus, the equivalent of fastai's WikiText-103 weights)."""
    model = get_language_model(len(dls.num.vocab), config, drop_mult)
    learn = TextLearner(dls, model, loss_func=CrossEntropyLossFlat(), splitter=awd_lstm_lm_split,
                        cbs=[ModelResetter(), RNNRegularizer(alpha=2, beta=1)], metrics=metrics, **kwargs)
    if pretrained and pretrained_fname is not None:
        state = torch.load(pretrained_fname, map_location=dls.device)
        model[0].load_state_dict(match_embeds(state["encoder"], state["vocab"], dls.num.vocab))
        learn.freeze()
    return learn


def text_classifier_learner(dls, arch=AWD_LSTM, config=None, drop_mult=0.5, metrics=None, pretrained=False,
                            pretrained_fname=None, **kwargs):
    """An AWD-LSTM classifier. In Chapter 10 the encoder comes from your fine-tuned language model
    (learn.load_encoder). pretrained=True instead loads a general-corpus encoder (Chapter 1's shortcut)."""
    model = get_text_classifier(len(dls.num.vocab), len(dls.vocab), config, drop_mult)
    learn = TextLearner(dls, model, loss_func=CrossEntropyLossFlat(), splitter=awd_lstm_clas_split,
                        cbs=[ModelResetter(), RNNRegularizer(alpha=2, beta=1)], metrics=metrics, **kwargs)
    if pretrained:
        cfg = {**awd_lstm_clas_config, **(config or {})}
        fname = pretrained_fname or pretrained_wikitext_lm(cfg)
        state = torch.load(fname, map_location=dls.device)
        model[0].module.load_state_dict(match_embeds(state["encoder"], state["vocab"], dls.num.vocab))
        learn.freeze()
    return learn


def pretrained_wikitext_lm(config, epochs=4, fname=None):
    """Our stand-in for fastai's WikiText-103 pretrained AWD-LSTM: train a language model on the
    synthetic 'wikitext' corpus once, save its encoder in data/models/, and reuse it afterwards."""
    from .data import URLs, untar_data
    from .learner import accuracy
    tag = "_".join(f"{config[k]}" for k in ("emb_sz", "n_hid", "n_layers"))
    fname = fname or DATA_ROOT / "models" / f"wikitext_synth_awd_lstm_{tag}.pth"
    if fname.exists():
        return fname
    print(f"[pretrain] training a language model on the synthetic WikiText corpus (first time only) -> {fname}")
    text = (untar_data(URLs.WIKITEXT, real=False) / "train.txt").read_text()
    docs = ["= " + d for d in text.split("\n= ") if d.strip()]          # one document per article
    dls = lm_dataloaders(docs, bs=64, seq_len=72, min_freq=2)
    learn = language_model_learner(dls, config=config, pretrained=False, metrics=[accuracy])
    learn.fit_one_cycle(epochs, 3e-3)
    fname.parent.mkdir(parents=True, exist_ok=True)
    learn.path = fname.parent
    learn.save_encoder(fname.stem)
    return fname
