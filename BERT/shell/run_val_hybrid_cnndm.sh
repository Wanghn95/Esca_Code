python train.py -task hybrid \
-mode validate -batch_size 15000 \
--hybrid_loss \
-control Rel \
-test_batch_size 15000 \
-bert_data_path /home/ybai/downloads/bert_data_cnndm_final/cnndm \
-log_file ../logs/val_hybrid_bert_hylossdec0002_ctrlrel07 \
-model_path /home/ybai/projects/PreSumm/Centrality_Pre_Summ/models_hybrid_cnndm_hylossdec0002_ctrlrel07 \
-sep_optim true \
-use_interval true -visible_gpus 2 \
-max_pos 512 -max_length 200 \
-alpha 0.95 -min_length 50 \
-result_path ../logs/hybrid_bert_cnndm_hylossdec0002_ctrlrel07 -test_all 
