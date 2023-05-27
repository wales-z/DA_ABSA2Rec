CUDA_VISIBLE_DEVICES=0 python main.py --per_gpu_train_batch_size 32 \
                --per_gpu_eval_batch_size 32 \
                --task_name musical_instruments \
                --learning_rate 2e-3\
                --weight_decay 1e-4 \
                --num_train_epochs 100 \
                --scheduler_gamma 0.9 \
                --seed 42