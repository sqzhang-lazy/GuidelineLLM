# GuidelineLLM

This is a code for the paper: Look Before You Leap: Enhancing Attention and Vigilance Regarding Harmful Content with GuidelineLLM


## Overview

This framework provides a comprehensive pipeline for generating jailbreak attacks against LLMs and creating defensive guidelines to prevent such attacks. The system consists of two main components:

1. **Jailbreak LLM**: Generates adversarial prompts to bypass LLM safety filters
2. **Guideline LLM**: Creates defensive guidelines to prevent jailbreak attacks

## Project Structure

```
.
├── gptGenerate/                    # Initial data preparation
│   ├── Attack_Template.json        # Descriptions of different jailbreak techniques
│   ├── guard_policy.json           # Harmful scenario descriptions
│   ├── template_get_attack_prompt.py          # Constructs attack queries
│   ├── template_attack2jailbreak_prompt.py    # Converts attack queries to jailbreaks
│   └── template_get_guideline_prompt.py       # Generates guidelines for jailbreaks
├── data/                           # Fine-tuning datasets
│   ├── jailbreak_query_for_jailbreak_finetune.json    # Jailbreak LLM training data
│   └── jailbreak_query_w_guideline_inference.json     # Guideline LLM training data
├── jailbreak_finetune/             # Jailbreak LLM fine-tuning code
├── guideline_finetune/             # Guideline LLM fine-tuning code
├── guideline_inference/            # Guideline inference and evaluation
└── jailbreak_inference/            # Jailbreak inference generation
```

## Data Formats

### Jailbreak LLM Training Data
**File:** `data/jailbreak_query_for_jailbreak_finetune.json`
```json
{
  "jailbreak_technique_name": [
    {
      "plain_attack": "original_harmful_query",
      "jailbreak": "generated_jailbreak_attack"
    }
  ]
}
```

### Guideline LLM Training Data
**File:** `data/jailbreak_query_w_guideline_inference.json`
```json
[
  {
    "jailbreak": "jailbreak_attack_text",
    "guideline": "corresponding_defensive_guideline"
  }
]
```


## Model Fine-tuning

### 1. Jailbreak LLM Fine-tuning
```bash
cd jailbreak_finetune
python train.py \
    --model "llama2-7B-chat" \
    --checkpoint_dir "llama2-7B-chat" \
    --input_path "data/jailbreak_query_for_jailbreak_finetune.json" \
    --save_dir "./checkpoints" \
    --epoch 3 \
    --lr 2e-5 \
    --warmup_ratio 0.03 \
    --input_len 1536 \
    --output_len 2560 \
    --batch_size 1 \
    --seed 42
```

### 2. Guideline LLM Fine-tuning
```bash
cd guideline_finetune
python train.py \
    --model "llama2-7B-chat" \
    --checkpoint_dir "llama2-7B-chat" \
    --input_path "data/jailbreak_query_w_guideline_inference.json" \
    --save_dir "./checkpoints" \
    --epoch 3 \
    --lr 2e-5 \
    --warmup_ratio 0.03 \
    --input_len 1536 \
    --output_len 2560 \
    --batch_size 1 \
    --seed 42
```

## Inference

### Guideline Generation and Evaluation

1. **Generate Guidelines for Jailbreaks:**
   - Edit `guideline_inference/run_submit_defendLLM_inference_guideline.sh`
   - Replace `checkpoint` with Guideline LLM model path
   - Modify `input_path` (list of jailbreak strings) and `output_path`
   - Pay attention to `llama2_7b_guideline_inference_wo_sys.py` line 121 which constructs the model input

```bash
cd guideline_inference
./run_submit_defendLLM_inference_guideline.sh
```

2. **Evaluate Guideline Effectiveness:**
   - Edit `guideline_inference/run_submit_guideline_against_jailbreak.sh`
   - Replace `model` with Defend LLM model path
   - Modify `input_path` (output from previous step) and `output_path`
   - Pay attention to the `guideline_against_jailbreak_inference.py` function `load_model_tokenizer` which loads your local model

```bash
cd guideline_inference
./run_submit_guideline_against_jailbreak.sh
```

### Jailbreak Generation

1. **Generate Jailbreak Attacks:**
   - Edit `jailbreak_inference/run_submit_jailbreakLLM_inference.sh`
   - Replace `checkpoint` with Jailbreak LLM model path
   - Set `jailbreak_set` with data in the same format as fine-tuning data

```bash
cd jailbreak_inference
./run_submit_jailbreakLLM_inference.sh
```

## Data Preparation

To prepare custom datasets:

1. **Attack Templates:** Define jailbreak techniques in `gptGenerate/Attack_Template.json`
2. **Harmful Scenarios:** Define attack scenarios in `gptGenerate/guard_policy.json`
3. **Generate Data:** Use the scripts in `gptGenerate/` to create initial training data:
   - `template_get_attack_prompt.py`: Creates attack queries
   - `template_attack2jailbreak_prompt.py`: Converts attacks to jailbreaks
   - `template_get_guideline_prompt.py`: Generates defensive guidelines


## Output Models

After training, you will obtain two specialized models:
1. **Jailbreak LLM:** Generates adversarial prompts
2. **Guideline LLM:** Creates defensive guidelines

## Notes

- Ensure sufficient GPU memory for 7B parameter models
- The framework uses LoRA or similar efficient fine-tuning techniques
- All scripts use random seed 42 for reproducibility
- Adjust batch size based on available GPU memory

## Applications

This framework can be used for:
- Red teaming LLM security
- Developing defensive mechanisms
- Research on adversarial attacks against language models
- Safety alignment testing


## Citation

If you use this framework in your research, please cite:
[Add citation information]
