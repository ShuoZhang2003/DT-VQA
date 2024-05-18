import pathlib
import textwrap
from argparse import ArgumentParser
import google.generativeai as genai
import json
from PIL import Image
from IPython.display import display
from IPython.display import Markdown
from tqdm import tqdm
import os

from eval import eval_anls, eval_accuracy, eval_accanls

def save_json(json_list,save_path):
    with open(save_path, 'w') as file:
        json.dump(json_list, file,indent=4)

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--prompt", type=bool, default=False)
    parser.add_argument("--DT-VQA_file", type=str, default="./data/test/DT_test.json")
    parser.add_argument("--output_folder", type=str, default="./results")
    parser.add_argument("--GOOGLE_API_KEY", type=str, default="")
    parser.add_argument("--model", type=str, default="gemini-pro-vision")
    parser.add_argument("--save_name", type=str, default="gemini")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _get_args()
    genai.configure(api_key=args.GOOGLE_API_KEY)
    model = genai.GenerativeModel(args.model)

    if os.path.exists(os.path.join(args.output_folder,f"{args.model}.json")):
        data_path = os.path.join(args.output_folder,f"{args.model}.json")
    else:
        data_path = args.DT-VQA_file
    
    with open(data_path, "r") as f:
        data = json.load(f)
    for i in tqdm(range(len(data))):
        img_path = data[i]['image_path']
        if args.prompt:
            prompt = "This is a question related to text in an image, please provide a word or phrase from the image text as an answer. "
            qs = prompt + data[i]['question']
        else:
            qs = data[i]['question']
        if data[i].get("predict", 0)!=0:
            print(f"{img_path} predict exist, continue.")
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            response = model.generate_content([qs, img])
            data[i]['predict'] = response.text
            save_json(data, os.path.join(args.output_folder,f"{args.model}.json"))
        except:
            print(f"{img_path}: API call failed.")
    
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
        if "predict" in data[i]:
            anls = eval_anls(data[i]["answers"], data[i]["predict"])
            accuracy = eval_accuracy(data[i]["answers"], data[i]["predict"])
            accanls = eval_accanls(data[i]["answers"], data[i]["predict"])
        else:
            anls = 0
            accuracy = 0
            accanls = 0
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