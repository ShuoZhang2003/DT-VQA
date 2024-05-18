import base64
import requests
from tqdm import tqdm
import json
from PIL import Image
import random
import time
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


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def save_json(json_list,save_path):
    with open(save_path, 'w') as file:
        json.dump(json_list, file,indent=4)

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--prompt", type=bool, default=False)
    parser.add_argument("--DT-VQA_file", type=str, default="./data/test/DT_test.json")
    parser.add_argument("--output_folder", type=str, default="./results")
    parser.add_argument("--OPENAI_API_KEY", type=str, default="")
    parser.add_argument("--model", type=str, default="gpt-4-vision-preview")
    parser.add_argument("--save_name", type=str, default="gpt4v")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _get_args()
    
    if os.path.exists(os.path.join(args.output_folder,f"{args.model}.json")):
        data_path = os.path.join(args.output_folder,f"{args.model}.json")
    else:
        data_path = args.DT-VQA_file

    with open(data_path,"r") as f:
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
        base64_image = encode_image(img_path)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.OPENAI_API_KEY}"
        }
        payload = {
            "model": args.model,
            "messages": [
              {
                "role": "user",
                "content": [
                  {
                    "type": "text",
                    "text": f"{qs}"
                  },
                  {
                    "type": "image_url",
                    "image_url": {
                      "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                  }
                ]
              }
            ],
            "max_tokens": 500
        }
        
        try:
          response = requests.post("https://api.closeai-asia.com/v1/chat/completions", headers=headers, json=payload)
          print(response.json())
          answer = response.json()['choices'][0]['message']['content']
          data[i]['predict'] = answer
          save_json(data, os.path.join(args.output_folder,f"{args.model}.json"))
        except:
          time.sleep(100)
          print(f"{img_path} error")
    
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
        if "predict" in (data[i]):
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