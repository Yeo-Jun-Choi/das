CUDA_VISIBLE_DEVICES=2 python main.py \
                                    --negative_num 128 \
                                    --dataset Yelp18_m1 \
                                    --gamma 1e-7 \
                                    --weight 1 \
                                    --learning_rate 1e-3 \
                                    --margin1 0.9 \
                                    --margin2 0.11 \
                                    --embedding_dim 512