# finetune_eval
Here are some files to evaluate LLM model based on lit-gpt(https://github.com/Lightning-AI/litgpt).

prepare_meadow.py: huggingfaceからmedalpaca/medical_meadow_medqaデータセットをダウンロードしてtraining datasetとtest datasetに分け、tokenizerにて変換・埋め込みされた後、指定の出力ディレクトリにtrain.pt, test.ptとして保存する。
