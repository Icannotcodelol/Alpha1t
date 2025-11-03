#!/bin/bash
# Stop training gracefully

echo "Stopping training..."
pkill -f "alpha_1000.rl.ppo_lstm.a2c_trainer"
sleep 2

if ps aux | grep "a2c_trainer" | grep -v grep > /dev/null; then
    echo "⚠️  Training still running, force killing..."
    pkill -9 -f "a2c_trainer"
else
    echo "✅ Training stopped successfully"
fi

echo ""
echo "Last progress:"
tail -5 training_ultimate.log
