export MODEL_DIR="path/to/FLUX.1-dev" #need to modity this in order to train your own model
export OUTPUT_DIR="outputs"
export TRAIN_DATA="dataset-sec-1024-high.jsonl"
export LOG_PATH="$OUTPUT_DIR/log"

accelerate launch --num_cpu_threads_per_process=1 PhotoDoodle-main/train_lora_flux_pe.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --width 1024 \
    --height 1024 \
    --source_column="source" \
    --target_column="target" \
    --caption_column="caption" \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --rank=64 \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --num_train_epochs=100 \
    --num_validation_images=2 \
    --validation_image "autodl-tmp/data-sec-con/1130-text-wb.png" \
    --validation_prompt "A purple teapot sits on a wooden table, beside a white teacup with steam rising from it; in the background, sunlight streams through a window, casting soft shadows. The <sks1> is in bold black characters, prominently displayed at the upper center of the image, while the teapot is oriented to the left, creating a warm and inviting atmosphere." \
    --validation_steps=1000 \
    --gradient_accumulation_steps=8 \
    --checkpointing_steps=300
    

