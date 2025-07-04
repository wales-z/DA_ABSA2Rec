#!/usr/bin/env bash
#!/usr/bin/env bash
TASK_NAME=laptop14
RS_TASK_NAME=cell_phones_and_accessories
ABSA_TYPE=tfm
CUDA_VISIBLE_DEVICES=0 python main.py --model_type bert \
                         --absa_type ${ABSA_TYPE} \
                         --tfm_mode finetune \
                         --fix_tfm 0 \
                         --model_name_or_path bert-base-uncased \
                         --data_dir ../data/${TASK_NAME} \
                         --task_name ${TASK_NAME} \
                         --rs_task_name ${RS_TASK_NAME} \
                         --rs_data_dir ../data/${RS_TASK_NAME} \
                         --per_gpu_train_batch_size 16 \
                         --per_gpu_eval_batch_size 8 \
                         --learning_rate 2e-5 \
                         --do_train \
                         --do_eval \
                         --do_lower_case \
                         --tagging_schema BIEOS \
                         --overfit 0 \
                         --overwrite_output_dir \
                         --eval_all_checkpoints \
                         --MASTER_ADDR localhost \
                         --MASTER_PORT 28512 \
                         --max_steps 1500
