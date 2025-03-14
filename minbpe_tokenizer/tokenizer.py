import json
import re
from collections import Counter
from typing import List, Iterable, Tuple, Any

import regex
from tqdm import tqdm

from . import data

class TokenizerInterface():

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Iterable[int]):
        raise NotImplementedError

    def train(self, text: str, vocab_size=4096, verbose=False) -> List[int]:
        raise NotImplementedError

    @classmethod
    def default_trained(cls, vocab_size=384, *args, **kwargs):
        tokenizer = cls(*args, **kwargs)
        tokenizer.train(data.training_text, vocab_size)
        return tokenizer


class Tokenizer(TokenizerInterface):

    START_BYTE = 256

    def __init__(self, vocab=None):
        # self._vocab = {i: i for i in range(256)}
        # to remove?
        # self._merges: dict = {}
        self._vocab: dict[int, Tuple[int, int]] = vocab if vocab else {}

    def save(self, file):
        with open(file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f)

    @classmethod
    def from_file(cls, file):
        with open(file, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        vocab = {int(idx): tuple(pair) for idx, pair in vocab.items()}
        return cls(vocab=vocab)

    def print_vocab(self):
        print("Vocab:")
        for _id, pair in self._vocab.items():
            try:
                b_str = b"".join(map(int.to_bytes, pair))
            except OverflowError as _:
                b_str = None
            decoded_pair = b_str.decode("utf-8", errors="replace") if b_str else "<complex pair>"
            print(f"Id: {_id}. Byte pair: {pair}. Decoded pair: {decoded_pair}")

    def __len__(self):
        # vocab size doesn't take into account 0x00 byte. Main vocab len = 255 (1-255)
        return self.START_BYTE + len(self._vocab)

class BasicTokenizer(Tokenizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_base_ids(self, merged_id, fully_decoded_id = None) -> List[int]:
        if fully_decoded_id is None:
            fully_decoded_id = []
        decoded_merge = self._vocab.get(merged_id)
        if decoded_merge is None:
            fully_decoded_id.append(merged_id)
            return fully_decoded_id
        for decoded_id in decoded_merge:
            self._get_base_ids(decoded_id, fully_decoded_id)

        return fully_decoded_id

    @staticmethod
    def find_freq_pair(ids: List[int]) -> (int, int):
        assert ids
        try:
            return Counter(zip(ids, ids[1:])).most_common(1)[0][0]
        except IndexError:
            return None

    @staticmethod
    def merge(ids: List[int], pair: (int, int), idx: int) -> Iterable[int]:
        assert len(ids) > 0
        if len(ids) == 1:
            return ids
        i = 1
        new_ids = []
        while i < len(ids):
            # if (ids[i - 1], ids[i]) == pair:
            if ids[i - 1] == pair[0] and ids[i] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i - 1])
                i += 1
        if new_ids[-1] != idx:
            new_ids.append(ids[-1])
        return new_ids

    def encode(self, text: str) -> List[int]:
        ids = list(text.encode("utf-8"))
        encoded = ids
        for _id, pair in self._vocab.items():
            encoded = BasicTokenizer.merge(encoded, pair, _id)
        return encoded

    def decode(self, ids: Iterable[int]):
        decoded_ids = []
        for new_id in ids:
            decoded_ids.extend(self._get_base_ids(new_id))
        b_str = b"".join(map(int.to_bytes, decoded_ids))
        return b_str.decode("utf-8", errors="replace")

    def train(self, text: str = data.training_text, vocab_size=4096, verbose=False) -> List[int]:
        assert vocab_size >= Tokenizer.START_BYTE
        ids = list(text.encode("utf-8"))
        for new_id in tqdm(range(Tokenizer.START_BYTE, vocab_size)):
            freq_pair = BasicTokenizer.find_freq_pair(ids)
            if not freq_pair:
                break
            self._vocab[new_id] = freq_pair
            # self._merges[freq_pair] = new_id
            ids = BasicTokenizer.merge(ids, freq_pair, new_id)
        if verbose:
           self.print_vocab()
        return ids

class RegexTokenizer(BasicTokenizer):

    GPT4_SPLIT_REGEX = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    def __init__(self, regex: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regex = regex if regex else self.GPT4_SPLIT_REGEX

    @staticmethod
    def merge(list_ids: List[List[int]], pair: (int, int), idx: int) -> Iterable[Iterable[int]]:
        return [BasicTokenizer.merge(split, pair, idx) for split in list_ids]

    @staticmethod
    def find_freq_pair(list_ids: List[List[int]]) -> (int, int):
        assert list_ids
        cnt = Counter()
        for split in list_ids:
            split_cnt = Counter(zip(split, split[1:]))
            cnt.update(split_cnt)
        try:
            return cnt.most_common(1)[0][0]
        except IndexError:
            return None

    def train(self, text: str = data.training_text, vocab_size=4096, verbose=False) -> List[int]:
        assert vocab_size >= self.START_BYTE
        splits = regex.findall(self.regex, text)
        ids = list(text.encode("utf-8"))
        list_ids = []
        for split in splits:
             list_ids.append(list(split.encode("utf-8")))
        for new_id in tqdm(range(self.START_BYTE, vocab_size)):
            freq_pair = self.find_freq_pair(list_ids)
            if not freq_pair:
                break
            self._vocab[new_id] = freq_pair
            # self._merges[freq_pair] = new_id
            list_ids = self.merge(list_ids, freq_pair, new_id)
        if verbose:
            self.print_vocab()
        return ids


class SpecialTokenizer(TokenizerInterface):

    PAD_TOKEN = "<pad>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"

    def __init__(self, tokenizer: Tokenizer):
        assert tokenizer
        self._tokenizer = tokenizer
        start_id = Tokenizer.START_BYTE + len(tokenizer._vocab)
        self._special_vocab = {start_id: self.PAD_TOKEN, start_id + 1: self.START_TOKEN, start_id + 2: self.END_TOKEN}
        self._special_vocab_inverted = {v: k for k, v in self._special_vocab.items()}

    @staticmethod
    def _split_text(text: str, separators: set[str]) -> list[str]:
        regex_sep = r"(" + r"|".join(separators) + r")"
        return [s for s in re.split(regex_sep, text) if s != ""]

    @staticmethod
    def _split_ids(ids: Iterable[int], separators: set[int]) -> list[list[int]]:
        list_of_list = []
        _list = []
        for _id in ids:
            if _id in separators:
                if _list:
                    list_of_list.append(_list)
                    _list = []
                list_of_list.append([_id])
            else:
                _list.append(_id)
        if _list:
            list_of_list.append(_list)
        return list_of_list

    def _encode(self, text: str) -> List[int]:
        separators = set(self._special_vocab.values())
        texts = self._split_text(text, separators)
        ids = []
        for text in texts:
            if text in separators:
                ids.append(self._special_vocab_inverted[text])
            else:
                ids.extend(self._tokenizer.encode(text))
        return ids

    def encode(self, text: str, start: bool = False, end: bool = False, pad: bool = False, max_len: int = 512):
        # max_len only matters if pad = True
        _ids = self._encode(text=text)
        ids = []
        if start and _ids[0] != self._special_vocab_inverted[self.START_TOKEN]:
            ids.append(self._special_vocab_inverted[self.START_TOKEN])
        ids.extend(_ids)
        if pad and ids[-1] != self._special_vocab_inverted[self.END_TOKEN]:
            pad_length = max_len - len(ids) - 1 if end else max_len - len(ids)
            if pad_length > 0:
                ids += pad_length * [self._special_vocab_inverted[self.PAD_TOKEN]]
        if end and ids[-1] != self._special_vocab_inverted[self.END_TOKEN]:
            ids.append(self._special_vocab_inverted[self.END_TOKEN])
        return ids

    def decode(self, ids: Iterable[int]):
        separators = set(self._special_vocab.keys())
        ids_list = self._split_ids(ids, separators)
        text = ""
        for ids in ids_list:
            if ids[0] in separators:
                text += self._special_vocab[ids[0]]
            else:
                text += self._tokenizer.decode(ids)
        return text

    def train(self, text: str, vocab_size=4096, verbose=False) -> List[int]:
        assert vocab_size >= Tokenizer.START_BYTE + len(self._special_vocab)
        for s in self._special_vocab.values():
            assert s not in text

        _vocab_size = vocab_size - len(self._special_vocab)
        ids = self._tokenizer.train(text=text, vocab_size=_vocab_size, verbose=verbose)
        # vocab can be smaller if training text is not enough to fill vocab
        special_vocab_start = len(self._tokenizer)
        new_special_vocab = {}
        for i, (_id, token) in enumerate(self._special_vocab.items()):
            new_special_vocab[special_vocab_start + i] = token
            self._special_vocab_inverted[token] = special_vocab_start + i
        self._special_vocab = new_special_vocab
        if verbose:
            print(self._special_vocab)
        return ids

    def __len__(self):
        return len(self._tokenizer) + len(self._special_vocab)
