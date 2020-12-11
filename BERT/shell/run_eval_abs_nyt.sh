python train.py -task abs -mode validate -test_all -model_path ../models_abs_nyt -bert_data_path /home/ybai/downloads/bert_data_nyt50_maxpos/nyt50 -dec_dropout 0.2  -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -test_batch_size 500 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -max_pos 512 -visible_gpus 2 -log_file ../logs/abs_bert_nyt_val -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_nyt_dir/nyt50
