from data import consts
import string


def txt2token(txt: string, vocab):
    token = [vocab.index(l) for l in txt]
    return token


def token2txt(token: list, vocab: dict):
    vocab_len = len(vocab)
    txt = []
    for index in token:
        if index >= vocab_len or index < 0:
            txt.append("*")
        else:
            txt.append(vocab[index])
    return txt

