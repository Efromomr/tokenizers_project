from tokenizers import NormalizedString, PreTokenizedString
import jieba
from MeCab import Tagger as JpTagger
from konlpy.tag import Kkma
import pyarabic.araby as araby
import textspan
from typing import Optional, List


class ChinesePreTokenizer:
    def jieba_split(self, i: int, normalized_string: NormalizedString):
        return [
            normalized_string[w[1] : w[2]]
            for w in jieba.tokenize(str(normalized_string))
        ]

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.jieba_split)


class JapanesePreTokenizer:
    def __init__(self, mecab_dict_path: Optional[str] = None):
        self.mecab = JpTagger("-Owakati")

    def tokenize(self, sequence: str) -> list[str]:
        return self.mecab.parse(sequence).strip().split(" ")

    def custom_split(
        self, i: int, normalized_string: NormalizedString
    ) -> list[NormalizedString]:
        text = str(normalized_string)
        tokens = self.tokenize(text)
        tokens_spans = textspan.get_original_spans(tokens, text)
        return [
            normalized_string[st:ed]
            for cahr_spans in tokens_spans
            for st, ed in cahr_spans
        ]

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.custom_split)


class KoreanPreTokenizer:
    def __init__(self):
        self.tokenizer = Kkma()

    def tokenize(self, sequence: str) -> List[str]:
        return self.tokenizer.morphs(sequence)

    def custom_split(
        self, i: int, normalized_string: NormalizedString
    ) -> List[NormalizedString]:
        text = str(normalized_string)
        tokens = self.tokenize(text)
        tokens_spans = textspan.get_original_spans(tokens, text)
        return [
            normalized_string[st:ed]
            for char_spans in tokens_spans
            for st, ed in char_spans
        ]

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.custom_split)


class ArabicPreTokenizer:
    def tokenizer(self, sequence: str):
        return araby.tokenize(sequence)

    def custom_split(self, i: int, normalized_string: NormalizedString):
        text = str(normalized_string)
        tokens = self.tokenizer(text)
        tokens_spans = textspan.get_original_spans(tokens, text)
        return [
            normalized_string[st:ed]
            for cahr_spans in tokens_spans
            for st, ed in cahr_spans
        ]

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.custom_split)
