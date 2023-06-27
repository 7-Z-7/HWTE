# HWTE
A implementation of HWTE from the paper "Hierarchical Wi-Fi Trajectory Embedding for Indoor User Mobility Pattern Analysis" in IMWUT 2023

## pretrain
'python3 run_pretraining_DR.py --input_file=./tf_dr_common_train_data.tfrecord --output_dir=./train_garden/pretraining_output --do_train=True --do_eval=True --bert_config_file=./bert_config_garden/rert_config_4.json --train_batch_size=32 --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=300000 --num_warmup_steps=10000 --learning_rate=1e-4 --sin_position=True --max_eval_steps=1000 --save_checkpoints_steps=20000 --gpu_device=0'

## finetune for classification task
python run_classifier_DR.py --task_name=leavebase --do_train=true --do_eval=true --data_dir=../../data/downstream_dataset/pickle_downstream/DR_downstream/leave_split_1_time --vocab_file=./ap_vocab.txt --bert_config_file=./bert_config_garden/rert_config_4.json --init_checkpoint=./train_garden/pretraining_output/model.ckpt-300000 --max_seq_length=32 --max_day_length=256 --train_batch_size=2 --eval_batch_size=2 --learning_rate=2e-5 --sin_position=True --stop_gradient=True --num_train_epochs=5.0 --output_dir=../../data/downstream_dataset/pickle_downstream/DR_downstream/leave_task --gradient_accumulation_multiplier=8 --gpu_device=7

## finetune for rank task
python3 run_rank_DR.py --task_name=nextbase --do_train=true --do_eval=true --data_dir=../../data/downstream_dataset/pickle_downstream/DR_downstream/next_split_1_time --vocab_file=./ap_vocab.txt --bert_config_file=./bert_config_garden/rert_config_4.json --init_checkpoint=./train_garden/pretraining_output/model.ckpt-300000 --max_seq_length=32 --max_day_length=28 --train_batch_size=8 --eval_batch_size=2 --learning_rate=2e-5 --sin_position=True --stop_gradient=True --num_train_epochs=10.0 --output_dir=../../data/downstream_dataset/pickle_downstream/DR_downstream/next_task --gradient_accumulation_multiplier=4 --predict_max_place=3 --gpu_device=0

## finetune for multivariate regression task
python3 run_schedule_DR.py --task_name=schedulebase --do_train=true --do_eval=true --data_dir=../../data/downstream_dataset/pickle_downstream/DR_downstream/schedule_split_1_time --vocab_file=./ap_vocab.txt --bert_config_file=./bert_config_garden/rert_config_4.json --init_checkpoint=./train_garden/pretraining_output/model.ckpt-300000 --max_seq_length=32 --max_day_length=28 --train_batch_size=8 --eval_batch_size=2 --learning_rate=2e-5 --sin_position=True --stop_gradient=True --num_train_epochs=10.0 --output_dir=../../data/downstream_dataset/pickle_downstream/DR_downstream/schedule_task --gradient_accumulation_multiplier=4 --regress_num=4 --gpu_device=1

