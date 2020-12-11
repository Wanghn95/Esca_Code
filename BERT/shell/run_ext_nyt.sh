python train.py --is_debugging -ext_layers 1 -model_path ../ext_model_nyt_layer1_pairwise -task ext -mode train -bert_data_path /home/ybai/downloads/bert_data_nyt50_maxpos/nyt50 -ext_dropout 0.1 -lr 2e-3 -visible_gpus 2 -report_every 50 -save_checkpoint_steps 1000 -batch_size 9000 -train_steps 100000 -accum_count 2 -log_file ../logs/ext_bert_nyt_1layer_pairwise -use_interval true -warmup_steps 10000 -max_pos 512
