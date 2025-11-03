#!/bin/bash
# Mixed-opponent training for better performance

cd /Users/maxhenkes/Desktop/Alpha1t
source .venv311/bin/activate

# Stop any existing training
pkill -f "a2c_trainer\|mixed_trainer" 2>/dev/null || true
sleep 2

# Start mixed-opponent training
# Resume from best 25k checkpoint
nohup python -m alpha_1000.rl.ppo_lstm.mixed_trainer \
    --iterations 25000 \
    --games-per-iter 8 \
    --batch-size 32 \
    --device cpu \
    --eval-every 250 \
    --eval-games 50 \
    --save-every 500 \
    --learning-rate 5e-5 \
    --entropy-coef 0.03 \
    --self-play-ratio 0.5 \
    --random-ratio 0.3 \
    --greedy-ratio 0.2 \
    --full-game \
    --resume runs/checkpoints_ultimate/agent_025000.pt \
    --logdir runs/tensorboard_mixed \
    --save-dir runs/checkpoints_mixed \
    > training_mixed.log 2>&1 &

PID=$!
echo "Mixed-opponent training started with PID: $PID"
echo "Resuming from: runs/checkpoints_ultimate/agent_025000.pt"
echo "Configuration: 50% self-play, 30% vs Random, 20% vs Greedy"
echo "Log file: training_mixed.log"
echo "Monitor with: tail -f training_mixed.log"
