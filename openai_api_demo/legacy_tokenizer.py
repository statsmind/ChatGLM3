import os
import re
from typing import List

from sentencepiece import SentencePieceProcessor


class LegacySPTokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.unk_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

        role_special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"]
        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"] + role_special_tokens
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1
        self.role_special_token_expression = "|".join([re.escape(token) for token in special_tokens]) # for apply_chat_template

    def tokenize(self, s: str, encode_special_tokens=True):
        if encode_special_tokens:
            last_index = 0
            t = []
            for match in re.finditer(self.role_special_token_expression, s):
                if last_index < match.start():
                    t.extend(self.sp_model.EncodeAsPieces(s[last_index:match.start()]))
                t.append(s[match.start():match.end()])
                last_index = match.end()
            if last_index < len(s):
                t.extend(self.sp_model.EncodeAsPieces(s[last_index:]))
            return t
        else:
            return self.sp_model.EncodeAsPieces(s)

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        text, buffer = "", []
        for token in t:
            if token in self.index_special_tokens:
                if buffer:
                    text += self.sp_model.decode(buffer)
                    buffer = []
                text += self.index_special_tokens[token]
            else:
                buffer.append(token)
        if buffer:
            text += self.sp_model.decode(buffer)
        return text

    def decode_tokens(self, tokens: List[str]) -> str:
        text = self.sp_model.DecodePieces(tokens)
        return text

    def convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.index_special_tokens:
            return self.index_special_tokens[index]
        if index in [self.eos_id, self.bos_id, self.pad_id] or index < 0 or index > self.sp_model.vocab_size():
            return ""
        return self.sp_model.IdToPiece(index)
