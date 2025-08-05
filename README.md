## Plane Geometry Diagram Formalization via Vision-Language Models

Code and data for the paper "Plane Geometry Diagram Formalization via Vision-Language Models".

[Paper](https://openreview.net/forum?id=RvUtaFyQZD), Dataset [GDF86K](https://huggingface.co/datasets/1509cxt/GDF86K), Model Checkpoints [Geo-TinyLLaVA](https://huggingface.co/1509cxt/Geo-TinyLLaVA).

Notice: Here we publish the model checkpoints of GeoTinyLLaVA trained with the Predicate Commutativity 
Transformation (PCT) augmented instruction-tuning dataset ([PCT.json](https://huggingface.co/datasets/1509cxt/GDF86K/blob/main/PCT.json)), which gives the best performance on Inter-GPS geometry problem solving among the 4 model checkpoints.


![ex1](assets/overview.png)

## Installation and Requirements

1. Clone this repository and navigate to the folder

```bash
git clone https://github.com/Geo-TinyLLaVA/Geo-TinyLLaVA.git
cd Geo-TinyLLaVA 
```

2. With Python 3.10+ and python3-pip, create and activate a virtualenv environment or conda environment, or just use your system Python interpreter.

3. Install all required python dependencies of TinyLLaVA and Inter-GPS.

```bash
# prerequisites of the training and inference of Geo-TinyLLaVA
cd TinyLLaVA_Factory 
pip install -e .         

# prerequisites of the evaluation of geometry diagram formalization performance and geometry problem solving performance of Inter-GPS
cd ..
pip install -r InterGPS/requirement.txt      
```
4. data download and unzip

```
cd InterGPS
. data/unzip_data.sh

cd ..
# download the dataset GDF86K
git clone https://huggingface.co/datasets/1509cxt/GDF86K
```

## Train Geo-TinyLLaVA from Scrach

```
cd TinyLLaVA_Factory
bash ./work/custom_finetune_PCT.sh
```

## Run Geo-TinyLLaVA 

```
cd TinyLLaVA_Factory
python ./work/inference_GeoTinyLLaVA.py

# combine the formalization and detection part to form a complete diagram parsing result
python ./work/combine_formalization_and_detection.py
```

## Geometry Diagram Formalization Performance Evaluation

```
cd InterGPS/diagram_parser/evaluation
python calc_diagram_accuracy.py \
--diagram_test /root/autodl-tmp/Geo-TinyLLaVA/TinyLLaVA_Factory/test_output/geotinyllava_pct/combined_diagram_parsing_result.json \
--diagram_gt ../../data/geometry3k/logic_forms/diagram_logic_forms_annot_refined.json
```

## Geometry Problem Solving Performance Evaluation

```
cd InterGPS/symbolic_solver
python test.py --label final_new \
--strategy final \
--text_logic_form_path ../text_parser/text_logic_forms.json \
--diagram_logic_form_path /root/autodl-tmp/Geo-TinyLLaVA/TinyLLaVA_Factory/test_output/geotinyllava_pct/combined_diagram_parsing_result.json \
--predict_path ../theorem_predict/results/test/pred_seqs_test_bart_best.json
```

## Acknowledgement and Citation
The project is built on top of the amazing [InterGPS](https://github.com/lupantech/InterGPS) and [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) repository. Thanks for their great work!

If you find our work helpful, please cite:
```
@misc{cui2025plane,
  title        = {Plane Geometry Diagram Formalization via {Vision-Language Models}},
  author       = {Cui, Xiaoteng and Liu, Yi},
  year         = {2025},
  month        = jul,
  howpublished = {the 2nd AI for {MATH} Workshop at the 42nd International Conference on Machine Learning ({ICML} 2025), Vancouver, Canada},
  note         = {Non-archival poster},
  url          = {https://openreview.net/forum?id=RvUtaFyQZD}
}
```
