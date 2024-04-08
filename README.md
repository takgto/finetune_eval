# finetune_eval
Here are some files to evaluate LLM model based on lit-gpt( https://github.com/Lightning-AI/litgpt ).  
必ずlit-gptの直下で以下のpython3コマンドを実行すること。  

**prepare_meadow.py:** huggingfaceからmedalpaca/medical_meadow_medqaデータセットをダウンロードしてtraining datasetとtest datasetに分け、tokenizerにて変換・埋め込みされた後、指定の出力ディレクトリにtrain.pt, test.ptとして保存する。lit-gpt/scriptsの下で実行する。
*python3 scripts/prepare_meadow.py*  
Option: 以下[]内はdefault値  
 --destination_path 出力ディレクトリ [/home/t_goto/lit-gpt/data/medical-meadow]  
 --test_split_fraction test datasetの比率 [0.03865]  

**my_generate.py:** finetuningしたモデルにtest_dataをfeedして回答を作る。  
*python3 finetune/my_generate.py*  
Option:  
 --max_new_tokens [20]  
 --llama2_dir 事前学習モデルのあるディレクトリ [/data2/goto_data/lit-gpt/checkpoints/meta-llama/Llama-2-7b-chat-hf]  
 --meadow_ckpt_dir ファインチューニングされたモデルのあるディレクトリ [/home/t_goto/lit-gpt/out/lora_merged/meadow]  
 --out_dir 回答のjsonファイルをstoreするディレクトリ [/home/t_goto/lit-gpt/out/comparison/meadow]  
 --data_dir test file (test.pt)のあるディレクトリ [/home/t_goto/lit-gpt/data/medical-meadow]  
 
**similarity_comp.py** my_generate.pyにより出力されるnoFT.json, FT.json, annotation.jsonファイルをから正解率を計算する。  
*python3 finetune/similarity_comp.py*  
Option:  
--out_dir my_generate.pyにより出力されるファイルのあるディレクトリ [/home/t_goto/lit-gpt/out/comparison/meadow]  
--threshold cos類似度で正解か不正解かを切り分ける閾値 [0.9]  
