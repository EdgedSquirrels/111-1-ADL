# ADL22-HW3

## Modification on transformers 4.22.2
The transformers I use in this homework is 4.22.2, with some modification on the file transformers/models/t5/modeling_t5.py to fix fp16 NAN issue. The modification is shown on this page: https://github.com/huggingface/transformers/compare/main...t5-fp16-no-nans

## Curves on report
To draw the learning curve on ROUGE, I evaluated the models saved on checkpoints. And draw the chart on Google Sheets. Due to the memory limitation, The checkpoints are not saved in the submission.

## Install tw_rouge
```bash
pip install -e code/tw_rouge
```

## Reproduce the result
```bash=
bash ./download.sh
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```

## Train
```bash=
python3.9 code/run_summarization-22.py \
    --model_name_or_path google/mt5-small \
    --cache_dir cache \
    --do_train \
    --train_file /path/to/train.jsonl \
    --source_prefix "summarize: " \
    --output_dir /path/to/output_dir \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --per_device_eval_batch_size=4 \
    --eval_accumulation_steps=4 \
    --predict_with_generate \
    --text_column maintext \
    --summary_column title \
    --validation_file /path/to/validation.jsonl \
    --optim adafactor \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --do_eval \
    --fp16 \
```

## Predict
```bash=
python3.9 run_summarization-22.py \
    --model_name_or_path /path/to/output_dir \
    --cache_dir cache \
    --do_predict \
    --num_beams=5 \
    --test_file /path/to/test.jsonl \
    --predict_file /path/to/predict.jsonl \
    --source_prefix "summarize: " \
    --output_dir /path/to/output_dir \
    --gradient_accumulation_steps=4 \
    --per_device_eval_batch_size=4 \
    --eval_accumulation_steps=4 \
    --predict_with_generate \
    --text_column maintext \
    --summary_column title \
    --fp16
```

## Reference
[cccntu/tw_rouge](https://github.com/cccntu/tw_rouge)
