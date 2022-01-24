CUDA_VISIBLE_DEVICES=0 python main.py --per_gpu_train_batch_size 256 \
                --per_gpu_eval_batch_size 256 \
                --learning_rate 2e-3\
                --weight_decay 1e-4 \
                --num_train_epochs 50 \
                --scheduler_gamma 0.9 \
                --seed 42