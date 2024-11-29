import math
import pandas as pd
from transformers import Trainer, TrainingArguments, BertForMaskedLM, DataCollatorForLanguageModeling
from tokenizers import Tokenizer
from alltoks.tokenizer_pretrain import pretrain_tokenizer
from data.dataset_preprocessing import preprocess_dataset
from models.bert_config import get_bert_config
from transformers import PreTrainedTokenizerFast
from tokenizers.processors import BertProcessing
from tokenizers.pre_tokenizers import BertPreTokenizer, PreTokenizer
from datasets import load_from_disk

def main(csv_path: str, lang: str, tokenizers: list[str] = ['wpe', 'bpe', 'unigram']):
    """
    Main function to train and evaluate tokenizers on a given CSV dataset.

    Parameters:
    ----------
    csv_path : str
        Path to the CSV file containing the dataset.
    lang : str
        Language code (e.g., 'en', 'ch', 'jp', 'ko', 'ar').
    tokenizers : list[str]
        List of tokenizer types to pretrain and evaluate (default: ['wpe', 'bpe', 'unigram']).
    update_progress : function
        Function to update the progress. Should be called as update_progress(progress).
    """

    if lang not in ['en', 'ch', 'jp', 'ko', 'ar']:
        raise Exception("Language is not supported. Possible language codes: en, ch, jp, ko, ar")

    scores_data = []

    # Основная часть работы — этапы токенизации
    for tokenizer_name in tokenizers:
        custom_tokenizer = False
        for _ in range(2):
            # Этап 1: Предобучение токенизатора
            tokenizer = pretrain_tokenizer(tokenizer_name, csv_path, pre_tokenizer=get_pre_tokenizer(lang) if custom_tokenizer else None)
            #tokenizer.post_processor = get_post_processor(tokenizer)

            #if custom_tokenizer:
            #    tokenizer.pre_tokenizer = PreTokenizer.custom(get_pre_tokenizer(lang))
            #else:
            tokenizer.pre_tokenizer = BertPreTokenizer()

            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                #unk_token='[UNK]',
                #cls_token='[CLS]',
                #sep_token='[SEP]',
                pad_token='[PAD]',
                mask_token='[MASK]'
            )

            if custom_tokenizer:
                tokenizer._tokenizer.pre_tokenizer = PreTokenizer.custom(get_pre_tokenizer(lang)())

           
            train_dataset, test_dataset = preprocess_dataset(tokenizer, csv_path)
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=0.15
            )

            model = BertForMaskedLM(get_bert_config())

            training_args = TrainingArguments(
                output_dir='./results',
                per_device_train_batch_size=8,
                num_train_epochs=1,
                save_steps=10000,
                save_total_limit=2,
                prediction_loss_only=True,
                fp16=True,
                report_to='none',
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=test_dataset
            )

            trainer.train()
            eval_results = trainer.evaluate()

            perplexity = math.exp(eval_results['eval_loss'])

            if custom_tokenizer:
                tokenizer_name += '_custom'

            scores_data.append([tokenizer_name, 'perplexity', perplexity])

            if lang == 'en':
                break

            custom_tokenizer = True

            
    scores_df = pd.DataFrame(scores_data, columns=['tokenizer', 'metric', 'score'])
    print(scores_df)

    return scores_df, train_dataset, test_dataset


def get_post_processor(tokenizer: Tokenizer):
    """
    Creates a post-processor for the tokenizer.
    """
    return BertProcessing(
        cls=("[CLS]", tokenizer.token_to_id('[CLS]')),
        sep=("[SEP]", tokenizer.token_to_id('[SEP]'))
    )


def get_pre_tokenizer(lang: str):
    from alltoks.custom_tokenizers import ChinesePreTokenizer, JapanesePreTokenizer, KoreanPreTokenizer, ArabicPreTokenizer
    
    pre_tokenizers = {
        'ch': ChinesePreTokenizer,
        'jp': JapanesePreTokenizer,
        'ko': KoreanPreTokenizer,
        'ar': ArabicPreTokenizer
    }

    return pre_tokenizers[lang]
