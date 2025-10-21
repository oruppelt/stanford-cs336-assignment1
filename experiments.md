BASELINE
python transformers_train.py --train_data ../artifacts/ts_train/train_tokens.npy --val_data ../artifacts/ts_train/val_tokens.npy --vocab_size 10000 --batch_size 64 --d_model 512 --num_layers 4 --num_heads 16 --context_length 256 --max_steps 20000 --device auto --experiment_name baseline64 --log_every 50 --eval_every 200 --max_lr 1.5e-3 --min_lr 1.5e-4

**LR TESTS**

Test 1
python transformers_train.py --train_data ../artifacts/ts_train/train_tokens.npy --val_data ../artifacts/ts_train/val_tokens.npy --vocab_size 10000 --batch_size 64 --d_model 512 --num_layers 4 --num_heads 16 --context_length 256 --max_steps 4000 --device auto --experiment_name baseline64_lrtest1 --log_every 50 --eval_every 200 --max_lr 5e-4 --min_lr 5e-5

Test 2
python transformers_train.py --train_data ../artifacts/ts_train/train_tokens.npy --val_data ../artifacts/ts_train/val_tokens.npy --vocab_size 10000 --batch_size 64 --d_model 512 --num_layers 4 --num_heads 16 --context_length 256 --max_steps 4000 --device auto --experiment_name baseline64_lrtest2 --log_every 50 --eval_every 200 --max_lr 7.5e-4 --min_lr 7.5e-5

Test 3
python transformers_train.py --train_data ../artifacts/ts_train/train_tokens.npy --val_data ../artifacts/ts_train/val_tokens.npy --vocab_size 10000 --batch_size 64 --d_model 512 --num_layers 4 --num_heads 16 --context_length 256 --max_steps 4000 --device auto --experiment_name baseline64_lrtest3 --log_every 50 --eval_every 200 --max_lr 1e-3 --min_lr 1e-4
 
Test 4
python transformers_train.py --train_data ../artifacts/ts_train/train_tokens.npy --val_data ../artifacts/ts_train/val_tokens.npy --vocab_size 10000 --batch_size 64 --d_model 512 --num_layers 4 --num_heads 16 --context_length 256 --max_steps 4000 --device auto --experiment_name baseline64_lrtest4 --log_every 50 --eval_every 200 --max_lr 1.5e-3 --min_lr 1.5e-4

Test 5
python transformers_train.py --train_data ../artifacts/ts_train/train_tokens.npy --val_data ../artifacts/ts_train/val_tokens.npy --vocab_size 10000 --batch_size 64 --d_model 512 --num_layers 4 --num_heads 16 --context_length 256 --max_steps 4000 --device auto --experiment_name baseline64_lrtest5 --log_every 50 --eval_every 200 --max_lr 2e-3 --min_lr 2e-4

Test 6
python transformers_train.py --train_data ../artifacts/ts_train/train_tokens.npy --val_data ../artifacts/ts_train/val_tokens.npy --vocab_size 10000 --batch_size 64 --d_model 512 --num_layers 4 --num_heads 16 --context_length 256 --max_steps 4000 --device auto --experiment_name baseline64_lrtest6 --log_every 50 --eval_every 200 --max_lr 3e-3 --min_lr 3e-4

Test 7
python transformers_train.py --train_data ../artifacts/ts_train/train_tokens.npy --val_data ../artifacts/ts_train/val_tokens.npy --vocab_size 10000 --batch_size 64 --d_model 512 --num_layers 4 --num_heads 16 --context_length 256 --max_steps 4000 --device auto --experiment_name baseline64_lrtest7 --log_every 50 --eval_every 200 --max_lr 7.5e-3 --min_lr 7.5e-4

Test 8
python transformers_train.py --train_data ../artifacts/ts_train/train_tokens.npy --val_data ../artifacts/ts_train/val_tokens.npy --vocab_size 10000 --batch_size 64 --d_model 512 --num_layers 4 --num_heads 16 --context_length 256 --max_steps 4000 --device auto --experiment_name baseline64_lrtest8 --log_every 50 --eval_every 200 --max_lr 1e-2 --min_lr 1e-3

**LR TEST RESULT**
============================================================
Experiment Summary: baseline64_lrtest1
============================================================
Total iterations: 81
Total time: 0.20 hours
Final train loss: 1.7354
Best train loss: 1.7354
Final val loss: 1.7180
Best val loss: 1.7180

============================================================
Experiment Summary: baseline64_lrtest2
============================================================
Total iterations: 81
Total time: 0.20 hours
Final train loss: 1.6680
Best train loss: 1.6680
Final val loss: 1.6488
Best val loss: 1.6488
============================================================

============================================================
Experiment Summary: baseline64_lrtest3
============================================================
Total iterations: 81
Total time: 0.20 hours
Final train loss: 1.6348
Best train loss: 1.6348
Final val loss: 1.6150
Best val loss: 1.6149

============================================================
Experiment Summary: baseline64_lrtest4
============================================================
Total iterations: 81
Total time: 0.20 hours
Final train loss: 1.6010
Best train loss: 1.6010
Final val loss: 1.5801
Best val loss: 1.5794

============================================================
Experiment Summary: baseline64_lrtest5
============================================================
Total iterations: 81
Total time: 0.20 hours
Final train loss: 1.5817
Best train loss: 1.5817
Final val loss: 1.5604
Best val loss: 1.5604

============================================================
Experiment Summary: baseline64_lrtest6
============================================================
Total iterations: 81
Total time: 0.20 hours
Final train loss: 1.5651
Best train loss: 1.5651
Final val loss: 1.5443
Best val loss: 1.5443

============================================================
Experiment Summary: baseline64_lrtest7
============================================================
Total iterations: 81
Total time: 0.20 hours
Final train loss: 1.5438
Best train loss: 1.5438
Final val loss: 1.5308
Best val loss: 1.5217

============================================================
Experiment Summary: baseline64_lrtest8
============================================================
Total iterations: 81
Total time: 0.21 hours
Final train loss: 2.0168
Best train loss: 2.0168
Final val loss: 1.9716
Best val loss: 1.9716

**Batch TESTS**

batch 128

python transformers_train.py --train_data ../artifacts/ts_train/train_tokens.npy --val_data ../artifacts/ts_train/val_tokens.npy --vocab_size 10000 --batch_size 128 --d_model 512 --num_layers 4 --num_heads 16 --context_length 256 --max_steps 10000 --device auto --experiment_name baseline128 --log_every 50 --eval_every 200 --max_lr 3e-3 --min_lr 3e-4

batch 256

python transformers_train.py --train_data ../artifacts/ts_train/train_tokens.npy --val_data ../artifacts/ts_train/val_tokens.npy --vocab_size 10000 --batch_size 256 --d_model 512 --num_layers 4 --num_heads 16 --context_length 256 --max_steps 5000 --device auto --experiment_name baseline256 --log_every 50 --eval_every 200 --max_lr 3e-3 --min_lr 3e-4


**Batch TEST RESULT**

============================================================
Experiment Summary: baseline64new
============================================================
Total iterations: 401
Total time: 0.97 hours
Final train loss: 1.3410
Best train loss: 1.3410
Final val loss: 1.3490
Best val loss: 1.3300

============================================================
Experiment Summary: baseline128
============================================================
Total iterations: 201
Total time: 1.01 hours
Final train loss: 1.3340
Best train loss: 1.3340
Final val loss: 1.3448
Best val loss: 1.3448

============================================================
Experiment Summary: baseline256
============================================================
Total iterations: 101
Total time: 1.13 hours
Final train loss: 1.3506
Best train loss: 1.3506
Final val loss: 1.3487
Best val loss: 1.3487

