TASK_NAME=laptop14
RS_TASK_NAME=cell_phones_and_accessories
ABSA_TYPE=tfm

echo 'ABSA dataset': ${TASK_NAME}
echo 'ABSA layer': ${ABSA_TYPE}
echo 'Rec dataset': ${RS_TASK_NAME}

# echo 'Now doing: train and save the BERT-ABSA network (including fine-tuning BERT and train ABSA layer'
# cd ../BERT_E2E_ABSA
# CUDA_VISIBLE_DEVICES=2 PYTHONIOENCODING=utf-8 python main.py --model_type bert \
#                          --absa_type ${ABSA_TYPE} \
#                          --tfm_mode finetune \
#                          --fix_tfm 0 \
#                          --model_name_or_path bert-base-uncased \
#                          --data_dir ../data/${TASK_NAME} \
#                          --task_name ${TASK_NAME} \
#                          --rs_task_name ${RS_TASK_NAME} \
#                          --rs_data_dir ../data/${RS_TASK_NAME} \
#                          --per_gpu_train_batch_size 8 \
#                          --per_gpu_eval_batch_size 2 \
#                          --learning_rate 2e-5 \
#                          --do_train \
#                          --do_eval \
#                          --do_lower_case \
#                          --tagging_schema BIEOS \
#                          --overfit 0 \
#                          --overwrite_output_dir \
#                          --eval_all_checkpoints \
#                          --MASTER_ADDR localhost \
#                          --MASTER_PORT 28512 \
#                          --max_steps 1500
# echo 'Done: train and save the BERT-ABSA network'

cd ../data
echo 'Now doing: preprocess the Rec dataset and split into train set and test set'
mkdir ${RS_TASK_NAME}
python preprocess.py --dataset ${RS_TASK_NAME}
echo 'Done: preprocess the Rec dataset and split into train set and test set'

echo 'Now doing: re-finetune BERT-ABSA and generate user/item document embeddings'
cd ../
CUDA_VISIBLE_DEVICES=3 python make_embedding.py --dataset ${RS_TASK_NAME} \
                         --fine_tune_learning_rate 2e-5 \
                         --per_gpu_batch_size 4 \
                         --max_step 2000 \
                         --reduced_embedding_size 128