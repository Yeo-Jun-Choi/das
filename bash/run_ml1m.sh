CUDA_VISIBLE_DEVICES=2 python main.py \
                                    --negative_num 32 \
                                    --dataset Movielens1M_m1 \
                                    --gamma 1e-7 \
                                    --weight 1.0 \
                                    --learning_rate 1e-3 \
                                    --margin1 1.3 \
                                    --margin2 0.11 \
                                    --embedding_dim 256