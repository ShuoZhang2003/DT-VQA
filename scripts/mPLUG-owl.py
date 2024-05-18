import json
from argparse import ArgumentParser
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import math
import multiprocessing
from multiprocessing import Pool, Queue, Manager

from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

from eval import eval_anls, eval_accuracy, eval_accanls


# https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl

def split_list(lst, n):
    length = len(lst)
    avg = length // n  # 每份的大小
    result = []  # 存储分割后的子列表
    for i in range(n - 1):
        result.append(lst[i*avg:(i+1)*avg])
    result.append(lst[(n-1)*avg:])
    return result

def save_json(json_list,save_path):
    with open(save_path, 'w') as file:
        json.dump(json_list, file,indent=4)

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--prompt", type=bool, default=False)
    parser.add_argument("--DT-VQA_file", type=str, default="./data/test/DT_test.json")
    parser.add_argument("--output_folder", type=str, default="./results")
    parser.add_argument("--model_path", type=str, default="./model_weights/mplug-owl")
    parser.add_argument("--save_name", type=str, default="mplug-owl")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    return args

def eval_worker(args, data, eval_id, output_queue):
    print(f"Process {eval_id} start.")
    pretrained_ckpt = args.model_path
    model = MplugOwlForConditionalGeneration.from_pretrained(
        pretrained_ckpt,
        torch_dtype=torch.bfloat16,
    )
    model.to(f"cuda:{eval_id}")
    image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
    tokenizer = MplugOwlTokenizer.from_pretrained(pretrained_ckpt)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    for i in tqdm(range(len(data))):
        img_path = data[i]['image_path']
        qs = data[i]['question']
        if args.prompt:
            print(args.prompt)
            qs = "This is a question related to text in an image, please provide a word or phrase from the image text as an answer. "+qs
        prompts = [
        f'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
        Human: <image>
        Human: {qs}
        AI: ''']
        if data[i].get("predict", 0)!=0:
            print(f"{img_path} predict exist, continue.")
            continue
        generate_kwargs = {
        'do_sample': False,
        'top_k': 1,
        'max_length': 100
        }
        images = [Image.open(img_path)]
        inputs = processor(text=prompts, images=images, return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = model.generate(**inputs, **generate_kwargs)
        sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True) 
        data[i]['predict'] = sentence
    output_queue.put({eval_id: data})
    print(f"Process {eval_id} has completed.")

if __name__=="__main__":
    multiprocessing.set_start_method('spawn')
    args = _get_args()
    
    if os.path.exists(os.path.join(args.output_folder,f"{args.save_name}.json")):
        data_path = os.path.join(args.output_folder,f"{args.save_name}.json")
        print(f"output_path:{data_path} exist! Only generate the results that were not generated in {data_path}.")
    else:
        data_path = args.DT-VQA_file

    with open(data_path, "r") as f:
        data = json.load(f)
    
    data_list = split_list(data, args.num_workers)
    output_queue = Manager().Queue()

    pool = Pool(processes=args.num_workers)
    for i in range(len(data_list)):
        pool.apply_async(eval_worker, args=(args, data_list[i], i, output_queue))
    pool.close()
    pool.join()
    
    results = {}
    while not output_queue.empty():
        result = output_queue.get()
        results.update(result)
        print("result not none")
    data = []
    for i in range(len(data_list)):
        data.extend(results[i])
    
    all_anls = []
    all_accuracy =[]
    all_accanls =[]
    DTNutrCap_anls = []
    DTNutrCap_accuracy = []
    DTNutrCap_accanls = []
    DTProdCap_anls = []
    DTProdCap_accuracy = []
    DTProdCap_accanls = []
    DTScene_anls = []
    DTScene_accuracy = []
    DTScene_accanls = []
    DTTabShot_anls = []
    DTTabShot_accuracy = []
    DTTabShot_accanls = []
    for i in range(len(data)):
        anls = eval_anls(data[i]["answers"], data[i]["predict"])
        accuracy = eval_accuracy(data[i]["answers"], data[i]["predict"])
        accanls = eval_accanls(data[i]["answers"], data[i]["predict"])
        data[i]["anls"] = anls
        data[i]["accuracy"] = accuracy
        data[i]["accanls"] = accanls
        all_anls.append(anls)
        all_accuracy.append(accuracy)
        all_accanls.append(accanls)
        if "POIE" in data[i]["type"]:
            DTNutrCap_anls.append(anls)
            DTNutrCap_accuracy.append(accuracy)
            DTNutrCap_accanls.append(accanls)
        if "DAST" in data[i]["type"]:
            DTProdCap_anls.append(anls)
            DTProdCap_accuracy.append(accuracy)
            DTProdCap_accanls.append(accanls)
        if "Hiertext" in data[i]["type"]:
            DTScene_anls.append(anls)
            DTScene_accuracy.append(accuracy)
            DTScene_accanls.append(accanls)
        if "Tabfact" in data[i]["type"]:
            DTTabShot_anls.append(anls)
            DTTabShot_accuracy.append(accuracy)
            DTTabShot_accanls.append(accanls)
    
    save_json(data, os.path.join(args.output_folder,f"{args.save_name}.json"))   
    
    avg_anls = sum(all_anls) / len(all_anls)
    avg_accuracy = sum(all_accuracy) / len(all_accuracy)
    avg_accanls = sum(all_accanls) / len(all_accanls)
    
    avg_DTNutrCap_anls = sum(DTNutrCap_anls) / len(DTNutrCap_anls)
    avg_DTNutrCap_accuracy = sum(DTNutrCap_accuracy) / len(DTNutrCap_accuracy)
    avg_DTNutrCap_accanls = sum(DTNutrCap_accanls) / len(DTNutrCap_accanls)
    
    avg_DTProdCap_anls = sum(DTProdCap_anls) / len(DTProdCap_anls)
    avg_DTProdCap_accuracy = sum(DTProdCap_accuracy) / len(DTProdCap_accuracy)
    avg_DTProdCap_accanls = sum(DTProdCap_accanls) / len(DTProdCap_accanls)
    
    avg_DTScene_anls = sum(DTScene_anls) / len(DTScene_anls)
    avg_DTScene_accuracy = sum(DTScene_accuracy) / len(DTScene_accuracy)
    avg_DTScene_accanls = sum(DTScene_accanls) / len(DTScene_accanls)
    
    avg_DTTabShot_anls = sum(DTTabShot_anls) / len(DTTabShot_anls)
    avg_DTTabShot_accuracy = sum(DTTabShot_accuracy) / len(DTTabShot_accuracy)
    avg_DTTabShot_accanls = sum(DTTabShot_accanls) / len(DTTabShot_accanls)
    
    print(f"avg_anls={avg_anls}")
    print(f"avg_accuracy={avg_accuracy}")
    print(f"avg_accanls={avg_accanls}")