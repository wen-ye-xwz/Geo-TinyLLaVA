import os
import json
from tqdm import tqdm



formalization_result_folder="./test_output/geotinyllava_pct"

# choose a source for the detection part of diagram_parsing_result, here we utilize the refined Ground Truth
detection_source_path="../InterGPS/data/geometry3k/logic_forms/diagram_logic_forms_annot_refined.json"

with open(detection_source_path, 'r') as file:
    detection_source = json.load(file)

diagram_parsing_result_output_path="./test_output/geotinyllava_pct/combined_diagram_parsing_result.json"

diagram_logic_form_dict = {}

for i in tqdm(range(2401,3002)):
    if os.path.exists(os.path.join(formalization_result_folder, str(i)+".txt"))==False:
        continue
    with open(os.path.join(formalization_result_folder, str(i)+".txt"), 'r') as file:
        elm=file.read()

    diagram_logic_form_dict[str(i)] = {}
    # split elm["text"] by sepatator ".", strip "space" for each element
    text_list = [x.strip() for x in elm.split(";") if x.strip()!=""]
    seen_set = set()
    literal_list = []
    for text in text_list:
        if text not in seen_set:
            seen_set.add(text)
            literal_list.append(text)
    diagram_logic_form_dict[str(i)]["point_instances"] = detection_source[str(i)]["point_instances"]
    diagram_logic_form_dict[str(i)]["point_positions"] = detection_source[str(i)]["point_positions"]
    diagram_logic_form_dict[str(i)]["line_instances"] = detection_source[str(i)]["line_instances"]
    diagram_logic_form_dict[str(i)]["circle_instances"] = detection_source[str(i)]["circle_instances"]
    diagram_logic_form_dict[str(i)]['diagram_logic_forms'] = literal_list
    

with open(diagram_parsing_result_output_path,"w") as f:
    f.write(json.dumps(diagram_logic_form_dict, indent=4))



