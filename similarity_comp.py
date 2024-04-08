import os
import sys
sys.path.append('/home/t_goto/lit-gpt')
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import CLI
from pathlib import Path
import numpy as np
from langchain.embeddings import OpenAIEmbeddings  #← OpenAIEmbeddingsをインポート
from numpy import dot  #← ベクトルの類似度を計算するためにdotをインポート
from numpy.linalg import norm  #← ベクトルの類似度を計算するためにnormをインポート
import copy
import itertools
import pandas as pd
import json

embeddings = OpenAIEmbeddings( #← OpenAIEmbeddingsを初期化する
    model="text-embedding-ada-002"
)

def common_substring(*strings):
    def same_all(*args):
        piv = args[0]
        return all([piv == e for e in args[1:]])
    return ''.join([s[0] for s in itertools.takewhile(lambda x: same_all(*x), zip(*strings))])

def comp_vector(ft, annot):
    ft_v = copy.deepcopy(ft)
    annot_v = copy.deepcopy(annot)
    #com_string = common_substring(ft_v, annot_v)
    #ft_v.replace(com_string,'')
    #annot_v.replace(com_string,'')
    q_vector = embeddings.embed_query(annot_v) #← 質問をベクトル化
    ft_vector = embeddings.embed_query(ft_v)
    cos_ft_ann = dot(q_vector, ft_vector) / (norm(q_vector) * norm(ft_vector))
    return cos_ft_ann

def similarity_comp(
    out_dir: Path = Path("/home/t_goto/lit-gpt/out/comparison/meadow"),
    save_result: bool = False,
    threshold: float = 0.9,
    debug_flag: bool = False,
) -> None:
    print(f'output directory: {out_dir}')
    #tokenizer = Tokenizer(checkpoint_dir)
    da = pd.read_json(out_dir / 'annotation.json')
    dF = pd.read_json(out_dir / 'FT.json')
    dFn = pd.read_json(out_dir / 'no_FT.json')
    dfc = pd.concat([dF['instruction'], dFn['output'], dF['output'], da['output']], axis='columns', ignore_index=True)
    dfc.columns = ['instruction', 'no_FT_out', 'FT_out', 'annotation']
    
    # for debug
    #if save_result:
    #    dfc.to_csv(out_dir / 'results.csv', index=False)

    no_ft_out = []
    ft_out = []
    si_no_ft = []
    si_ft = []
    judge = []
    scores = []
    score_idx = []
    score_ft = []
    score_ann = []
    for k, (no_ft, ft, annot) in enumerate(zip(dfc['no_FT_out'], dfc['FT_out'], dfc['annotation'])):
        if k>=1000:
            break

        # remove 'com_string' between ft and noft, and also remove '\n'
        com_string = common_substring(ft, no_ft)
        ft = ft.replace(com_string,'')
        ft = ft.replace('\n',' ')
        ft_out.append(ft)

        no_ft = no_ft.replace(com_string,'')
        no_ft = no_ft.replace('\n', ' ')
        no_ft_out.append(no_ft)

        # calculate similarity
        #print(f'common_string={com_string}')
        print(f'--- {k} ---')
        print(f'no_ft={no_ft}')
        similar_no_ft = comp_vector(annot, no_ft)
        si_no_ft.append(similar_no_ft)

        print(f'ft={ft}')
        similar_ft = comp_vector(annot, ft)
        si_ft.append(similar_ft)

        if similar_ft >= threshold:
            score_idx.append(k)
            scores.append(similar_ft)
            score_ft.append(ft)
            score_ann.append(annot)

        judge.append(1 if similar_no_ft < similar_ft else 0)
            
    dfout = pd.DataFrame({'no_ft_out':no_ft_out, 'ft_out':ft_out, 'similarity_noFT':si_no_ft, 'similarity_FT':si_ft, 'FT>noFT':judge})
    dfout.columns = ['no_ft_out', 'ft_out', 'similarity_noFT', 'similarity_FT', 'FT>noFT']
    
    df = pd.concat([dfc, dfout], axis='columns')

    if debug_flag:
        df_dbg = pd.DataFrame({'index':score_idx, 'score':scores, 'score_ft':score_ft, 'score_ann':score_ann})
        df_dbg.columns = ['index', 'score', 'FT', 'annotation']
        df_dbg.to_csv(out_dir / 'debug.csv', index=False)

    if save_result:
        df.to_csv(out_dir / (out_dir.name + '_result.csv'), index=False)

    # calculation of score
    score = 0
    for similarity in df['similarity_noFT']:
        if similarity >= threshold:
            score = score + 1
    print('### w/o Fine Tuning ###')
    print(f'score={score}')
    print(f'score={score/len(dfc)*100.0:.2f}%\n')

    score = 0
    for similarity in df['similarity_FT']:
        if similarity >= threshold:
            score = score + 1
    print('### w/ Fine Tuning ###')
    print(f'score={score}')
    print(f'score={score/len(dfc)*100.0:.2f}%\n')

if __name__ == "__main__":

    CLI(similarity_comp)

