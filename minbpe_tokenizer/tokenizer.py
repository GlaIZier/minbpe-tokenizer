from collections import Counter
from typing import List, Iterable

from tqdm import tqdm


class Tokenizer:

    def __init__(self):
        # self._vocab = {i: i for i in range(256)}
        self._vocab = {}
        self._merges = {}

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
        return Counter(zip(ids, ids[1:])).most_common(1)[0][0]

    @staticmethod
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

    def encode(self, text: str) -> List[int]:
        ids = list(text.encode("utf-8"))
        encoded = ids
        for _id, pair in self._vocab.items():
            encoded = Tokenizer.merge(encoded, pair, _id)
        return encoded

    def decode(self, ids: Iterable[int]):
        decoded_ids = []
        for new_id in ids:
            decoded_ids.extend(self._get_base_ids(new_id))
        b_str = b"".join(map(int.to_bytes, decoded_ids))
        return b_str.decode("utf-8", errors="replace")

    def train(self, text: str, vocab_size=4096, verbose=False) -> List[int]:
        assert vocab_size > 255
        ids = list(text.encode("utf-8"))
        for new_id in tqdm(range(256, vocab_size)):
            freq_pair = Tokenizer.find_freq_pair(ids)
            self._vocab[new_id] = freq_pair
            self._merges[freq_pair] = new_id
            ids = Tokenizer.merge(ids, freq_pair, new_id)
        if verbose:
            print("Vocab:")
            for _id, pair in self._vocab.items():
                try:
                    b_str = b"".join(map(int.to_bytes, pair))
                except OverflowError as _:
                    b_str = None
                decoded_pair = b_str.decode("utf-8", errors="replace") if b_str else "<complex pair>"
                print(f"Id: {_id}. Byte pair: {pair}. Decoded pair: {decoded_pair}")
        return ids
