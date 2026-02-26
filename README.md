Пайплайн для оценки токенизаторов, для интерфейса используется Flask.

Для локального запуска: `python app.py`

Для хостинга в Google Colab с помощью ngrok запустите `tokenization_pipeline.ipynb`.

Примеры датасетов для запуска: `arabic_dataset.csv`, `japanese_dataset.csv`, `korean_dataset.csv`. Собственные наборы данных должны быть с расширением .csv и иметь колонку 'text'.

Tokenization evaluation pipeline (Flask is used for the interface part)

Running with cli: `python app.py`   

Running with ngrok: `tokenization_pipeline.ipynb`

Input data examples: `arabic_dataset.csv`, `japanese_dataset.csv`, `korean_dataset.csv`. Custom datasets should be in .csv files with a 'text' column.


