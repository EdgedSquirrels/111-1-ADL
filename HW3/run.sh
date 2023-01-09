python3.9 code/run_summarization-22.py \
    --model_name_or_path summarization_model \
    --cache_dir cache \
    --do_predict \
    --num_beams=5 \
    --test_file ${1} \
    --predict_file ${2} \
    --source_prefix "summarize: " \
    --output_dir summarization_model \
    --gradient_accumulation_steps=4 \
    --per_device_eval_batch_size=4 \
    --eval_accumulation_steps=4 \
    --predict_with_generate \
    --text_column maintext \
    --summary_column title \
    --fp16

# bash ./run.sh data/public.jsonl data/pred.jsonl