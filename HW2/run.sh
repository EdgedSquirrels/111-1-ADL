

python3.9 multiple-choice/run_swag.py \
  --model_name_or_path models_bert/multiple-choice \
  --cache_dir cache \
  --do_predict \
  --learning_rate 5e-5 \
  --output_dir models_bert/multiple-choice \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size=2 \
  --test_file "${2}" \
  --predict_file intermediate.json \
  --context_file "${1}" \
  --pad_to_max_length False \
  --max_seq_length=512

python3.9 question-answering/run_qa.py \
  --cache_dir cache \
  --context_file "${1}" \
  --test_file intermediate.json \
  --predict_file "${3}" \
  --model_name_or_path models_bert/question-answering \
  --do_predict \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-5 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir models_bert/question-answering \
  --pad_to_max_length False



# bash ./download.sh
# bash ./run.sh /path/to/context.json /path/to/test.json  /path/to/pred/prediction.csv
# "${1}": path to the context file.
# "${2}": path to the testing file.
# "${3}": path to the output predictions.
# CUDA_VISIBLE_DEVICES=2,3,5 bash ./run.sh context.json test.json prediction.csv