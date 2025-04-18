import argparse
import os 
import numpy as np
import torch
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from importlib.metadata import version

from datasets import load_dataset
from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers, prune_admm, prune_prox, prune_APM, prune_mAPM
from lib.eval import eval_ppl, eval_zero_shot

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_llm(model_name, cache_dir="llm_weights"):
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # You can also set load_in_4bit if needed
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto",
        # quantization_config=quantization_config
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", "admm", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search","prox","APM","mAPM"])
    parser.add_argument("--step", type=float, default=0.001)
    parser.add_argument("--iteration", type=int, default=20)
    
    parser.add_argument("--cache_dir", default="/scratch/llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    

    model.seqlen = model.config.max_position_embeddings 
    model.eval()
    print(model.hf_device_map)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)


    start_time = time.time()
    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "admm":
            prune_admm(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "prox":
            prune_prox(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,alpha=args.step,iteration=args.iteration)
        elif args.prune_method == "APM":
            prune_APM(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,alpha=args.step,iteration=args.iteration)
        elif args.prune_method == "mAPM":
            prune_mAPM(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,alpha=args.step,iteration=args.iteration)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("running time")
    print(elapsed_time)

    ###############################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")
    print(f"time {elapsed_time}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}_{args.sparsity_ratio}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test\ttime(s)", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}\t{elapsed_time:.4f}", file=f, flush=True)


    ###################################################################

    if args.eval_zero_shot:
        accelerate=True
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"]
        # task_list = ["boolq","arc_challenge","arc_easy"]

        # task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge"]
        num_shot = 0
  
        
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)
         

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()
