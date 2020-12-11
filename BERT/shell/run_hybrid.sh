python train.py --oracle -temp_dir=../temp -lr_bert 0.002 -lr_dec 0.2 -task hybrid -mode train -model_path=../models_hybrid_cnndm -bert_data_path /home/ybai/downloads/bert_data_cnndm_final/cnndm -ext_dropout 0.1 -visible_gpus 2 -report_every 50 -save_checkpoint_steps 2000 -batch_size 500 -train_steps 200000 -accum_count 5  -log_file ../logs/hybrid_bert_cnndm -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512