
from tinyllava.eval.batch_eval_tinyllava import eval_model
import os
from tqdm import tqdm
import json
import re

model_path = "../checkpoints/geotinyllava_pct/checkpoint-86080"

conv_mode = "phi"
formalization_output_dir="./test_output/geotinyllava_pct"

if not os.path.exists(formalization_output_dir):
    os.makedirs(formalization_output_dir)

test_json_data_path = "/root/autodl-tmp/Geo-TinyLLaVA/TinyLLaVA_Factory/work/test_data_in_converation.json"

image_folder_dir = "../InterGPS/data"


args = type('Args', (), {
    "model_path": model_path,
    "model": None,
    "eval_data_file": test_json_data_path,
    "conv_mode": conv_mode,
    "image_folder_dir": image_folder_dir,
    "eval_output_dir": formalization_output_dir,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 4,
    "max_new_tokens": 1024

})()


eval_model(args)