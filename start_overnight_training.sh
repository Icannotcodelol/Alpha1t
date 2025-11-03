#!/bin/bash
# Ultimate overnight training with all improvements

cd /Users/maxhenkes/Desktop/Alpha1t
source .venv311/bin/activate

# Clean up old stale processes first
pkill -f "alpha_1000.rl.ppo_lstm.trainer" 2>/dev/null || true
pkill -f "alpha_1000.rl.ppo_lstm.a2c_trainer" 2>/dev/null || true
sleep 2

# Start training with proven stable configuration
nohup python -m alpha_1000.rl.ppo_lstm.a2c_trainer \
    --iterations 25000 \
    --curriculum default \
    --batch-size 32 \
    --device cpu \
    --eval-every 250 \
    --save-every 500 \
    --learning-rate 5e-5 \
    --entropy-coef 0.03 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --max-grad-norm 0.5 \
    --logdir runs/tensorboard_ultimate \
    --save-dir runs/checkpoints_ultimate \
    > training_ultimate.log 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo "Log file: training_ultimate.log"
echo "Monitor with: tail -f training_ultimate.log"
echo "Stop with: kill $PID"
