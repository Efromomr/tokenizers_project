def load_dataset_from_txt(file_path: str):
    """
    Reads data from .txt file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()