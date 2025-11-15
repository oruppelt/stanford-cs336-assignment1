python transformers_train.py --train_data ../artifacts/ts_train/train_tokens.npy --val_data ../artifacts/ts_train/val_tokens.npy --vocab_size 10000 --batch_size 16 --context_length 512 --max_steps 100 --device auto

python transformers_train.py --train_data ../artifacts/ts_train/train_tokens.npy --val_data ../artifacts/ts_train/val_tokens.npy --vocab_size 10000 --batch_size 16 --context_length 256 --max_steps 1000 --device auto --experiment_name baseline --log_every 50 --eval_every 500

python generate_text.py --checkpoint checkpoints/final.pt --config checkpoints/config.json --vocab ../artifacts/ts_train/vocab.json --merges ../artifacts/ts_train/merges.json