python train.py  \
-task hybrid -mode test -batch_size 4000 \
--hybrid_loss \
-test_batch_size 500 \
-test_from ../models_hybrid_cnndm_hylossdec0002_ctrlrel07/model_step_24000.pt \
-bert_data_path /home/ybai/downloads/bert_data_cnndm_final/cnndm \
-log_file ../logs/val_hybrid_bert_cnndm_ctrlrel07.log \
-model_path ../models_hybrid_cnndm_hyconnect/ \
-use_interval true -visible_gpus 5 \
-max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 \
-result_path ../logs/hybrid_bert_cnndm_hylossdec0002_ctrlrel07

# --hybrid_loss \
