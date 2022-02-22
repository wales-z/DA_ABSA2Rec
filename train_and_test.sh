CUDA_VISIBLE_DEVICES=1 python main.py --per_gpu_train_batch_size 16 \
                --per_gpu_eval_batch_size 16 \
                --task_name automotive \
                --learning_rate 1e-3\
                --weight_decay 1e-4 \
                --num_train_epochs 50 \
                --scheduler_gamma 0.9 \
                --seed 42