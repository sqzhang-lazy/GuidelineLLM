python3 llama2_7b_jailbreak_inference.py \
    --model vicuna-7B \
    --batch_size 1 \
    --checkpoint  \
    --mode turn2 \
    --jailbreak_set  \
    --generate_num 3600 \
    --max_length 4096