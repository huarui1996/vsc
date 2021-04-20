from tabulate import tabulate
import bcolz
import numpy as np
import os
import pickle
from torch import nn
import spacy
import torch


def generate_vocab_rawtext(dataset):
    sentence = []
    for i in range(len(dataset)):
        _, text = dataset[i]
        sentence.append(text)
    vocab = set()
    vocab.add("#EOS")
    vocab.add("#SOS")
    vocab.add("#PAD")
    for sen in sentence:
        words = str(sen).split(" ")
        for word in words:
            vocab.add(word.lower())
    vocab.remove("")
    raw_text = ""
    for sen in sentence:
        raw_text += sen.rstrip() + '. '
    return vocab, raw_text[:-3]


def verbose(cmd, config):
    print("Current Command is %s" % cmd)
    vsc = config['vsc']
    cbow = config['cbow']
    loss_conf = config['loss']
    if cmd == 'train':
        data = [(v, k) for v, k in dict(loss_conf).items()]
        print("Loss Setting:")
        print(tabulate(data, tablefmt='fancy_grid'))
    if cmd != 'cbow' and cmd != 'vocab':
        data = [(v, k) for v, k in dict(vsc).items()]
        print("VSC Model Setting:")
        print(tabulate(data, tablefmt='fancy_grid'))
    data = [(v, k) for v, k in dict(cbow).items()]
    print("CBOW Model Setting:")
    print(tabulate(data, tablefmt='fancy_grid'))


def glove_txt_to_pkl(glove_path, out_dir, dim=300):
    import os
    words = []
    idx = 0
    word2idx = {}
    print("hello")
    vectors = bcolz.carray(np.zeros(1), rootdir=os.path.join(out_dir,
                                                             'glove.6B.%d.dat'%dim), mode='w')

    with open(glove_path, 'rb') as f:
        for l in f:
            line = l.decode().split()
            print(line)
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
    vectors = bcolz.carray(vectors[1:].reshape((-1, dim)), rootdir=os.path.join(out_dir,
                                                                                    'glove.6B.%d.dat'%dim),
                           mode='w')
    vectors.flush()
    pickle.dump(words, open(os.path.join(out_dir, 'glove.6B.%d_words.pkl' % dim), 'wb'))
    pickle.dump(word2idx, open(os.path.join(out_dir, 'glove.6B.%d_idx.pkl' % dim), 'wb'))


def load_glove(glove_path, return_glove=False, dim=300):
    vectors = bcolz.open(os.path.join(glove_path, 'glove.6B.%d.dat' % dim))[:]
    vectors = np.append(vectors, np.zeros(shape=(1, dim), dtype=np.float), axis=0)
    words = pickle.load(open(os.path.join(glove_path, 'glove.6B.%d_words.pkl' % dim), 'rb'))
    words.append("#PAD")
    word2idx = pickle.load(open(os.path.join(glove_path, 'glove.6B.%d_idx.pkl' % dim), 'rb'))
    word2idx["#PAD"] = len(words) - 1
    if return_glove:
        glove = {w: vectors[word2idx[w]] for w in words}
        return glove, word2idx
    return vectors, words, word2idx


def decode_sequence(ix_to_word, seq):
    if isinstance(seq, torch.Tensor):
        seq = seq.cpu()
    else:
        seq = torch.tensor(seq)
    #print("seq", seq)
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j].item()
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out


class RewardCriterion(nn.Module):

    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.contiguous().view(-1)
        reward = reward.contiguous().view(-1)
        mask = (seq > 0).float()
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1).cuda(),
                         mask[:, :-1]], 1).contiguous().view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.loss_fn = nn.NLLLoss(reduce=False)

    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        # truncate to the same size
        batch_size = logits.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output


if __name__ == '__main__':
    glove_txt_to_pkl(glove_path='G:\\glove.6B.50d.txt', out_dir='G:\\glove-50d', dim=50)
    """
    spacy_en = spacy.load('en_core_web_sm')
    def tokenizer(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]
    a = tokenizer("I'm a student in BJFU.")
    a.append('#PAD')
    #Text = torchtext.data.Field(sequential=True, tokenize=tokenizer, lower=True)
    print(a)
    glove, w2i = load_glove("G:\\", return_glove=True)
    print(glove['#PAD'])
    print(glove['eos'])
    print(glove["#PAD"].shape)
    print(glove['the'].shape)
    print(type(glove['the']))
    print(w2i['unk'])
    print(len(w2i))
    """