python train.py \
--hybrid_loss \
-train_from ../models_hybrid_cnndm_hyloss1/model_step_16000.pt \
-control Rel \
-sep_optim false \
-lr 0.002 \
-task hybrid \
-mode train -model_path=../models_hybrid_cnndm_hylossdec0002_ctrlrel05 \
-bert_data_path /home/ybai/downloads/bert_data_cnndm_final/cnndm \
-ext_dropout 0.1 -visible_gpus 0 -report_every 50 \
-save_checkpoint_steps 2000 -batch_size 3500 \
-train_steps 200000 -accum_count 5  \
-log_file ../logs/hybrid_bert_cnndm_hylossdec0002_ctrlrel05 \
-use_interval true \
-warmup_steps 5000 \
-max_pos 512 
# -warmup_steps_bert 5000 -warmup_steps_dec 5000 \
# -lr_bert 0.002 -lr_dec 0.002 \
# -train_from ../models_hybrid_cnndm_hyloss3/model_step_20000.pt \
# -train_from_extractor ../ext_model_cnndm_layer1/model_step_29000.pt \
# -train_from_abstractor /home/ybai/downloads/model_step_148000.pt \
