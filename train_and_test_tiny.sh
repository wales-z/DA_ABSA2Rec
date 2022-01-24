CUDA_VISIBLE_DEVICES=2 python main.py --per_gpu_train_batch_size 16 \
                --per_gpu_eval_batch_size 16 \
                --learning_rate 3e-3\
                --weight_decay 1e-4 \
                --num_train_epochs 100 \
                --seed 42 \
                --scheduler_gamma 0.9 \
                --tiny