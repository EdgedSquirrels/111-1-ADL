# ADL HW2


## Reproduce the result
```bash=
bash ./download.sh
bash ./run.sh /path/to/context.json /path/to/test.json /path/to/prediction.csv
```

## Run the Bonus code for hw1:
You may visit `bonus` for more information.
```bash=
cd bonus
```

## Fine-tune the pre-trained model again
For the following commands, it will re-load the pretrained model and fine-tune again. `train_file`, `validation_file`, `test_file`, `predict_file` and `context_file` should be set by your choice.

### Train Multiple Choice:
```bash=
python3.9 multiple-choice/run_swag.py \
  --model_name_or_path bert-base-chinese \
  --cache_dir cache \
  --do_train \
  --do_eval \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --output_dir models_bert/multiple-choice \
  --gradient_accumulation_steps 8 \
  --per_gpu_train_batch_size=2 \
  --per_device_train_batch_size=2 \
  --per_device_eval_batch_size=2 \
  --overwrite_output \
  --train_file path/to/train.json \
  --validation_file path/to/valid.json \
  --context_file path/to/context.json \
  --pad_to_max_length False \
  --max_seq_length=512
```


### Test Multiple Choice:
```bash=
python3.9 multiple-choice/run_swag.py \
  --model_name_or_path models_bert/multiple-choice \
  --cache_dir cache \
  --do_predict \
  --learning_rate 5e-5 \
  --output_dir models_bert/multiple-choice \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size=2 \
  --test_file path/to/test.json \
  --predict_file path/to/pred.json \
  --context_file path/to/context.json \
  --pad_to_max_length False \
  --max_seq_length=512
```

### Train Question Answering:
```bash=
python question-answering/run_qa.py \
  --cache_dir cache \
  --context_file path/to/context.json \
  --train_file path/to/train.json \
  --validation_file path/to/valid.json \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --do_train \
  --do_eval \
  --gradient_accumulation_steps 8 \
  --per_gpu_train_batch_size=2 \
  --per_device_train_batch_size=2 \
  --per_device_eval_batch_size=2 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir models_bert/question-answering \
  --overwrite_output \
  --pad_to_max_length False \
  --logging_steps 400
```

### Test Question Answering:
```bash=
python question-answering/run_qa.py \
  --cache_dir cache \
  --context_file path/to/context.json \
  --test_file path/to/pred.json \
  --predict_file path/to/pred.csv \
  --model_name_or_path models_bert/question-answering \
  --do_predict \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-5 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir models_bert/question-answering \
  --pad_to_max_length False
```

