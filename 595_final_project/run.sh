CUDA_VISIBLE_DEVICES=0 python run_mlm.py \
    --dataset_name wikipedia \
    --dataset_config_name 20200501.en \
    --model_name_or_path bert-base-uncased \
    --output_dir checkpoints/test