# HW2 Bonus

In addition the the packages in HW2, `seqeval` package is needed.
```bash
pip install seqeval
```

For the following commands, `train_file` and `validation_file` should be chosen by your choice.

## Intent Classification

```bash
python3.9 run_glue.py \
  --cache_dir cache \
  --model_name_or_path bert-base-cased \
  --train_file path/to/intent/train.json \
  --validation_file path/to/intent/eval.json \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir models_bert/text-classification \
  --overwrite_output_dir
```

## Slot Tagging
```bash
python3.9 run_ner.py \
  --cache_dir cache \
  --model_name_or_path bert-base-uncased \
  --train_file path/to/slot/train.json \
  --validation_file path/to/slot/eval.json \
  --output_dir models_bert/token-classification \
  --do_train \
  --do_eval \
  --overwrite_output_dir
```
