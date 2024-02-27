#!/usr/bin/env bash

model_path=split3-10shot-run2-
scrpath=/mnt/1tra/hajizadeh/yolov5_obb/runs/val/split3-run2-10shot-pcb-

# ------------------------------- save-json ---------------------------------- #
python val.py --weights /mnt/1tra/hajizadeh/yolov5_obb/runs/train/${model_path}/weights/best.pt  --batch-size 16 --data data/dotav15_poly_fewshot.yaml --name ${scrpath} --save-json  --exist-ok

# ------------------------------- TestJson2VocClassTxt ---------------------------------- #
python tools/TestJson2VocClassTxt.py --json_path ${scrpath}/best_obb_predictions.json --save_path ${scrpath}/obb_predictions_Txt

# ------------------------------- ResultMerge ---------------------------------- #
python DOTA_devkit/ResultMerge.py --scrpath ${scrpath}/obb_predictions_Txt --dstpath ${scrpath}/obb_predictions_Txt_Merged

# ------------------------------- dota_evaluation_task1 ---------------------------------- #
python DOTA_devkit/dota_evaluation_task1.py --detpath ${scrpath}/obb_predictions_Txt_Merged/Task1_{:s}.txt
