import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Literal
import json

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies import FSDPStrategy
from torchmetrics.aggregation import RunningMean

# support running without installing as a package
wd = Path('/home/t_goto/lit-gpt')
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt.model import GPT, Block, Config
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    CLI,
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    get_default_supported_precision,
    load_checkpoint,
    num_parameters,
)
from scripts.prepare_alpaca import generate_prompt
from lightning.fabric.plugins import BitsandbytesPrecision

def base_t(
    prompt: str = "What food do llamas eat?",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: Optional[int] = 200,
    temperature: float = 0.8,
    model: nn.Module = None,
    tokenizer: Tokenizer = None,
    fabric: L.Fabric = None,
    #quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    #precision: Optional[str] = None,
    compile: bool = False,
) -> None:
    """Generates text samples based on a pre-trained model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_dir: The checkpoint directory to load.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        precision: Indicates the Fabric precision setting to use.
        compile: Whether to compile the model.
    """
    if model is None:
        print('Model is not provided!')
        return -1
    if tokenizer is None:
        print('Tokenizer is not provided!')
        return -1

    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    with fabric.init_tensor():
        model.max_seq_length = max_returned_tokens
        model.set_kv_cache(batch_size=1)

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
        t = time.perf_counter() - t0
        for block in model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        output = tokenizer.decode(y)
        fabric.print(output)
        tokens_generated = y.size(0) - prompt_length
        fabric.print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr
        )
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)

    return output

def load_model(
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    precision: Optional[str] = None,
    #compile: bool = False,
) -> Tuple[nn.Module, Tokenizer, L.Fabric]:
    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(checkpoint_dir / "lit_config.json")

    checkpoint_path = checkpoint_dir / "lit_model.pth"

    tokenizer = Tokenizer(checkpoint_dir)

    #fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    #fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    #fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    #encoded = tokenizer.encode(prompt, device=fabric.device)
    #prompt_length = encoded.size(0)
    #max_returned_tokens = prompt_length + max_new_tokens
    '''
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    '''
    model.eval()
    '''
    if compile:
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        global next_token
        next_token = torch.compile(next_token, mode="reduce-overhead")
    '''
    model = fabric.setup_module(model)

    #t0 = time.perf_counter()
    load_checkpoint(fabric, model, checkpoint_path)

    return model, tokenizer, fabric

def json_make(path: Path, obj: dict) -> None:
    ls = None
    with open(path, 'r+') as f:
        ls = f.readlines()
        if ls == []:
            ls.append('[\n')
        if ls[-1] == ']':
            ls[-1] = ','
        ls.insert(len(ls), f'{json.dumps(obj, indent=4 ,ensure_ascii=False)}')
        ls.insert(len(ls), '\n]')

    with open(path, 'w') as f:
        f.writelines(ls)

def json_preprocess(out_json: Path):
    if out_json.exists():
        out_json.unlink(missing_ok=False)
        out_json.touch()
    out_json.touch()

def setup(
    precision: Optional[str] = None,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"]] = None,
    devices: int = 1,
    seed: int = 1337,
    max_new_tokens: int = 20,
    llama2_dir: Path = Path("/data2/goto_data/lit-gpt/checkpoints/meta-llama/Llama-2-7b-chat-hf"),
    meadow_ckpt_dir: Path = Path("/home/t_goto/lit-gpt/out/lora_merged/meadow_q"),
    out_dir: Path = Path("/home/t_goto/lit-gpt/out/comparison/meadow_q"),
    data_dir: Path = Path("/home/t_goto/lit-gpt/data/medical-meadow"),
) -> None:
    # didplay parameters
    print(f'precision: {precision}')
    print(f'quantize: {quantize}')
    print(f'max_new_tokens = {max_new_tokens}')
    print(f'pretrain_model_dir: {llama2_dir}')
    print(f'lora_ckpt_dir: {meadow_ckpt_dir}')
    print(f'output_dir: {out_dir}')

    model_llama, tokenizer_llama, fabric_llama = load_model(llama2_dir)
    model_test, tokenizer_test, fabric_test = load_model(meadow_ckpt_dir)

    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    
    out_json1 = out_dir / 'no_FT.json'
    json_preprocess(out_json1)


    out_json2 = out_dir / 'FT.json'
    json_preprocess(out_json2)

    annotation = out_dir / 'annotation.json'
    json_preprocess(annotation)

    test_data = torch.load( data_dir / "test.pt")
    for k, data in enumerate(test_data):
        if (k < 1000):
            dict1 = {}
            dict2 = {}
            annot = {}
            print(f'########## prompt:{k} ##########')
            #prompt = data["instruction"] + " " + data["input"]
            prompt = data["instruction"] + " " + data["input"]

            out1 = base_t(prompt=prompt, max_new_tokens=max_new_tokens, model=model_llama, 
                          tokenizer=tokenizer_llama, fabric=fabric_llama) 
            # original model
            out2 = base_t(prompt=prompt, max_new_tokens=max_new_tokens, model=model_test, 
                          tokenizer=tokenizer_test, fabric=fabric_test) 
            # fine tuned model        
        
            dict1['instruction'] = dict2['instruction'] = annot['instruction'] = prompt

            prompt_mod = out1.split(sep='\n')[0]
            dict1['output'] = out1.replace(prompt_mod, '')
            prompt_mod = out2.split(sep='\n')[0]
            dict2['output'] = out2.replace(prompt_mod, '')
            annot['output'] = data['output']
        
            json_make(out_json1, dict1)
            json_make(out_json2, dict2)
            json_make(annotation, annot)

            #del out1, out2
            #torch.cuda.empty_cache()

        k = k + 1

if __name__ == "__main__":

    CLI(setup)

