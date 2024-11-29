import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer, PreTokenizer
from tokenizers.normalizers import BertNormalizer
from tokenizers.processors import TemplateProcessing
from datasets import Dataset
from typing import Optional
import typing

models_dict = {'wpe': WordPiece, 'bpe': BPE, 'unigram': Unigram}
trainers_dict = {'wpe': WordPieceTrainer, 'bpe': BpeTrainer, 'unigram': UnigramTrainer}

def pretrain_tokenizer(tokenizer_name: str, csv_path: Optional[str] = None, pre_tokenizer: Optional[PreTokenizer] = None):
    """
    Pretrains a tokenizer on the given CSV dataset and saves it to a .json file.

    Parameters
    ----------
    tokenizer_name : str
        A string to get classes for instantiating and training a tokenizer.
    csv_path : str
        A path to a CSV file containing the dataset for training the tokenizer.
    pre_tokenizer : tokenizers.pre_tokenizers.PreTokenizer
        A custom PreTokenizer for specific language.
    """

    def get_training_corpus() -> typing.Iterator[str]:
        """Yields 1000 lines of text from the dataset."""
        for i in range(0, len(dataset), 1000):
            yield dataset[i:i + 1000]["text"]

    
    if csv_path is None:
        raise ValueError("A valid CSV file path must be provided.")
    
    df = pd.read_csv(csv_path, encoding='utf-8')
    
   
    if "text" not in df.columns:
        raise ValueError("The CSV file must contain a 'text' column.")
    
    dataset = Dataset.from_pandas(df)

    
    tokenizer = Tokenizer(models_dict[tokenizer_name]())
    tokenizer.normalizer = BertNormalizer(lowercase=True)

    
    if pre_tokenizer is not None:
        tokenizer.pre_tokenizer = PreTokenizer.custom(pre_tokenizer())
    else:
        tokenizer.pre_tokenizer = BertPreTokenizer()

    
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers_dict[tokenizer_name](vocab_size=25000, special_tokens=special_tokens)

    
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

   
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )

    return tokenizer

   