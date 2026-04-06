# ==============================================================================
# Copyright (c) 2024 Tiange Luo, tiange.cs@gmail.com
# Last modified: September 04, 2024
#
# This code is licensed under the MIT License.
# ==============================================================================
"""
This script is used to caption the images using OpenAI API.

Usage:
    python captioning_openai.py --api_key <your_api_key> \
        --output_file <output_file> \
        --parent_dir <parent_dir> \
        --verified_uids <verified_uids> \
        --base_caption <base_caption>
"""

OPENAI_API_KEY = "xxxxxxxxxxxxxxx"

import base64
import requests
import os
import numpy as np
import pandas as pd
import glob
import argparse
from IPython import embed
from utils import *
import json
from PIL import Image
import shutil
import time
from tqdm import tqdm
logger = get_logger("logger")

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

parser = argparse.ArgumentParser(description="Process API key and json file path.")
parser.add_argument('--api_key', type=str, default=OPENAI_API_KEY, help="Your OpenAI API Key.")
parser.add_argument('--output_file', type=str, default='../lumitex_prompts/prompt_verified.json', help="Path to the output JSON file.")
parser.add_argument('--parent_dir', type=str, default='../lumitex', help="Path to the parent directory.")
parser.add_argument('--verified_uids', type=str, default='../lumitex_prompts/verified_uids_all.txt', help="Path to the verified uids json file.")
parser.add_argument('--base_caption', type=str, default='../lumitex_prompts/prompt_3dtopia.json', help="Path to the a base caption file.")

args = parser.parse_args()

api_key = args.api_key
output_file = args.output_file
verified_uids = open(args.verified_uids, 'r').read().splitlines()
base_caption = json.load(open(args.base_caption, 'r'))

captions_dict = {
    uid: base_caption[uid] for uid in verified_uids if base_caption.get(uid) is not None
}

caption_num = len(verified_uids) - len(captions_dict)
logger.info(f"Find {len(verified_uids)} verified uids with {len(captions_dict)} have captions, captioning {caption_num} uids")

# paths = glob.glob(os.path.join(args.parent_dir, '*'))

wrong_or_none_files = []
responses = {}
processed_num = 0
total_tokens = 0
for uid in tqdm(verified_uids):
    if captions_dict.get(uid) is not None:
        continue
    
    processed_num += 1
    
    image_paths = []

    for view_id in (1, 5, 3, 7):
        image_paths.append(os.path.join(args.parent_dir, f'{uid}/hdr/rgba/{(view_id+8):03d}.png'))

    base64_images = []
    tmp_dir = '.tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    
    # move the images to the tmp_dir
    os.makedirs(os.path.join(tmp_dir, uid), exist_ok=True)
    for view_id in (1, 5, 3, 7):
        # copy the image to the tmp_dir
        img = Image.open(os.path.join(args.parent_dir, f'{uid}/hdr/rgba/{(view_id+8):03d}.png'))
        img = img.resize((512, 512))
        img.save(os.path.join(tmp_dir, f'{uid}/{view_id:03d}.png'))
    continue
    
    try:
        for p in image_paths:
            # first resize the image to 512x512
            img = Image.open(p)
            img = img.resize((512, 512))
            img.save(os.path.join(tmp_dir, os.path.basename(p)))
            base64_images.append(encode_image(os.path.join(tmp_dir, os.path.basename(p))))
    except Exception as e:
        wrong_or_none_files.append(uid)
        logger.error(e)
        continue
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        # "model": "gpt-4o-2024-05-13",
        # "model": "gpt-4o-mini-2024-07-18",
        "model": "gpt-4.1-mini-2025-04-14",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Renderings show different angles of the same set of 3D objects. Concisely describe 3D object (distinct features, material, etc) as a caption, not mentioning angles and image related words. Suggested no more than 30 words"
                    },
                ]
            }
        ],
        "max_completion_tokens": 300
    }
    for i in range(len(base64_images)):
        payload['messages'][0]['content'].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_images[i]}"
                }
            }
        )
    
    try:
        response = requests.post("https://api.openai-proxy.com/v1/chat/completions", headers=headers, json=payload)
    except Exception as e:
        logger.error(e)
        continue
    try:
        r = response.json()
    except Exception as e:
        logger.error(e)
        continue
    responses[uid] = r
    try:
        curr_caption = r['choices'][0]['message']['content']
        total_tokens += r['usage']['total_tokens']
    except Exception as e:
        logger.error(e)
        continue
    
    # logger.debug(r)
    captions_dict[uid] = curr_caption
    logger.info(f"[{processed_num}/{caption_num}]: {uid} - {curr_caption}")

with open(output_file, 'w') as f:
    json.dump(captions_dict, f, indent=4)

logger.success(f"Captions saved to {output_file}")
logger.info(f"Total tokens: {total_tokens}")
