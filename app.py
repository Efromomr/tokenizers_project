from flask import Flask, render_template, request, send_file, jsonify, session
import pandas as pd
import tempfile
from main import main
import os
from pyngrok import ngrok

app = Flask(__name__)
app.secret_key = os.urandom(24)

example_file_path = "data/chinese_small.csv"


@app.route("/")
def home():
    articles = [
        {
            "title": "Challenging Language-Dependent Segmentation for Arabic: An Application to Machine Translation and Part-of-Speech Tagging",
            "link": "https://arxiv.org/abs/1709.00616",
        },
        {
            "title": "Evaluating Various Tokenizers for Arabic Text Classification",
            "link": "https://arxiv.org/abs/2106.07540",
        },
        {
            "title": "Exploring Tokenization Strategies and Vocabulary Sizes for Enhanced Arabic Language Models",
            "link": "https://arxiv.org/abs/2403.11130",
        },
        {
            "title": "Vaporetto: Efficient Japanese Tokenization Based on Improved Pointwise Linear Classification",
            "link": "https://arxiv.org/abs/2406.17185",
        },
        {
            "title": "How do different tokenizers perform on downstream tasks in scriptio continua languages?: A case study in Japanese",
            "link": "https://arxiv.org/abs/2306.09572",
        },
        {
            "title": "Improving Korean NLP Tasks with Linguistically Informed Subword Tokenization and Sub-character Decomposition",
            "link": "https://arxiv.org/abs/2311.03928",
        },
        {
            "title": "Korean-English Machine Translation with Multiple Tokenization Strategy",
            "link": "https://arxiv.org/abs/2105.14274",
        },
        {
            "title": "Sub-Character Tokenization for Chinese Pretrained Language Models",
            "link": "https://arxiv.org/abs/2106.00400",
        },
    ]

    return render_template("index.html", articles=articles)


@app.route("/tokenize", methods=["GET", "POST"])
def tokenize():
    result = None
    perplexity = None
    preview_data = None
    train_df = None
    test_df = None
    file_path = None

    if request.method == "POST":
        lang = request.form.get("language")
        file = request.files.get("file")
        action = request.form.get("action")

        if action == "example":
            df = pd.read_csv(example_file_path)
            results_df, train_df, test_df = main(csv_path=example_file_path, lang="ch")
            preview_data = results_df

        elif action == "preview":
            if not file:
                return render_template(
                    "tokenize.html", error="Please upload a valid CSV file."
                )
            file_path = tempfile.mktemp(suffix=".csv")
            file.save(file_path)

            results_df, train_df, test_df = main(csv_path=file_path, lang=lang)
            preview_data = results_df

        elif action == "download":
            if not file:
                return render_template(
                    "tokenize.html", error="Please upload a valid CSV file."
                )

            file_path = tempfile.mktemp(suffix=".csv")
            file.save(file_path)

            results_df, train_df, test_df = main(csv_path=file_path, lang=lang)

            merged_df = pd.concat(
                [train_df.to_pandas(), test_df.to_pandas()], ignore_index=True
            )

            output_path = tempfile.mktemp(suffix=".csv")
            merged_df.to_csv(output_path, index=False)

            return send_file(
                output_path,
                as_attachment=True,
                download_name="merged_tokenized_data.csv",
            )

    return render_template(
        "tokenize.html", result=result, perplexity=perplexity, preview_data=preview_data
    )


if __name__ == "__main__":
    public_url = ngrok.connect(5000)
    print(f"Сервер доступен по адресу: {public_url}")
    app.run()
