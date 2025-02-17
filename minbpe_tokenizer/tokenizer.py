from collections import Counter
from typing import List, Iterable, Sized


def find_freq_pair(ids: List[int]) -> (int, int):
    return Counter(zip(ids, ids[1:])).most_common(1)[0][0]

def merge(ids: List[int], pair: (int, int), idx: int) -> Iterable[int]:
    i = 1
    new_ids = []
    while i < len(ids):
        if (ids[i - 1], ids[i]) == pair:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i - 1])
            i += 1
    if new_ids[-1] != idx:
        new_ids.append(ids[-1])
    return new_ids

def merge_seq(ids: List[int], vocab_size=276):
    new_ids = ids.copy()
    for i in range(256, vocab_size):
        new_ids = merge(new_ids, find_freq_pair(new_ids), i)
    return new_ids

def get_merges(ids: List[int], vocab_size=276):
    new_ids = ids.copy()
    merges = {}
    for i in range(256, vocab_size):
        freq_pair = find_freq_pair(new_ids)
        merges[freq_pair] = i
        new_ids = merge(new_ids, freq_pair, i)
    return merges


def unmerge_seq(new_ids, merges):
    vocab = {i: (i,) for i in range(256)}
    for m, new_id in merges.items():
        vocab[new_id] = m

    def _get_id(new_id):
        _fully_decoded_id = []

        def __get_ids(__new_id):
            __decoded_merge = vocab[__new_id]
            if len(__decoded_merge) == 1:
                _fully_decoded_id.append(__new_id)
                return
            for __decoded_id in __decoded_merge:
                __get_ids(__decoded_id)

        __get_ids(new_id)
        return _fully_decoded_id

    decoded_ids = []
    for new_id in new_ids:
        decoded_ids.extend(_get_id(new_id))
    return decoded_ids


def decode_unmerged(unmerged_new_ids):
    b_str = b"".join(map(int.to_bytes, unmerged_new_ids))
    return b_str.decode("utf-8", errors="replace")
import pytest

def decode(new_ids, merges):
    return decode_unmerged(unmerge_seq(new_ids, merges))


class Tokenizer:

    def __init__(self):
        # self._vocab = {i: i for i in range(256)}
        self._vocab = {}
        self._merges = {}

    def encode(self, text: str) -> List[int]:
        ids = list(text.encode("utf-8"))
        encoded = ids
        for _id, pair in self._vocab.items():
            encoded = merge(encoded, pair, _id)
        return encoded

    def decode(self, ids: Iterable[int]):

        pass

    def train(self, text: str, vocab_size=4096):
        assert vocab_size > 255
        ids = list(text.encode("utf-8"))
        for i in range(256, vocab_size):
            freq_pair = find_freq_pair(ids)
            self._vocab[i] = freq_pair
            self._merges[freq_pair] = i
            ids = merge(ids, freq_pair, i)
