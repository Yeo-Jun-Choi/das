CUDA_VISIBLE_DEVICES=2 python main.py \
                                    --negative_num 256 \
                                    --dataset Gowalla_m1 \
                                    --gamma 1e-7 \
                                    --weight 0.25 \
                                    --learning_rate 1e-3 \
                                    --margin1 0.5 \
                                    --margin2 0.1 \
                                    --embedding_dim 512