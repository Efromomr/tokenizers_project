import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from datasets import Dataset

def preprocess_dataset(tokenizer: Tokenizer, csv_path: str, test_size: float = 0.3):
    """
    Preprocesses a CSV dataset using a tokenizer, splits it into train and test datasets, and returns them.

    Parameters
    ----------
    tokenizer : Tokenizer
        An instance of a tokenizer to preprocess the dataset.
    csv_path : str
        Path to the CSV file containing the dataset.
    test_size : float
        Proportion of the dataset to include in the test split (default: 0.3).

    Returns
    -------
    train_dataset : pd.DataFrame
        Tokenized training dataset.
    test_dataset : pd.DataFrame
        Tokenized test dataset.
    """

    
    df = pd.read_csv(csv_path)

    if "text" not in df.columns:
        raise ValueError("The CSV file must contain a 'text' column.")

    dataset = Dataset.from_pandas(df)

    dataset = dataset.train_test_split(test_size=0.3)

    for s in ['train', 'test']:
        dataset[s] = dataset[s].map(
            lambda samples: tokenizer(samples["text"],
                                      max_length=512,
                                      padding="max_length",
                                      truncation=True),
            #num_proc=10, batched=True
        )

    return dataset['train'], dataset['test']
