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

from transformers import TextStreamer
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from eval import eval_anls, eval_accuracy, eval_accanls

# https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2

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
    parser.add_argument("--model_path", type=str, default="./model_weights/mplug-owl2")
    parser.add_argument("--save_name", type=str, default="mplug-owl2")
    parser.add_argument("--conv_mode", type=str, default="mplug_owl2")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    return args

def eval_worker(args, data, eval_id, output_queue):
    print(f"Process {eval_id} start.")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, model_name, load_8bit=False, load_4bit=False, device=f"cuda:{eval_id}")
    for i in tqdm(range(len(data))):
        img_path = data[i]['image_path']
        if args.prompt:
            print(args.prompt)
            prompt = "This is a question related to text in an image, please provide a word or phrase from the image text as an answer. "
            qs = prompt + data[i]['question']
        else:
            qs = data[i]['question']
        if data[i].get("predict", 0)!=0:
            print(f"{img_path} predict exist, continue.")
            continue
        conv = conv_templates[args.conv_mode].copy()
        roles = conv.roles
        image = Image.open(img_path).convert('RGB')
        max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))
        image_tensor = process_images([image], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        inp = DEFAULT_IMAGE_TOKEN + qs
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=args.temperature,
                max_new_tokens=100,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()    
        data[i]['predict'] = outputs
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