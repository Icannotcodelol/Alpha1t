#!/bin/bash
# Monitor training progress

echo "=== TRAINING STATUS ==="
echo ""

# Check if training is running
PROCESS=$(ps aux | grep "a2c_trainer.*25000" | grep -v grep)
if [ -n "$PROCESS" ]; then
    PID=$(echo "$PROCESS" | awk '{print $2}')
    CPU=$(echo "$PROCESS" | awk '{print $3}')
    MEM=$(echo "$PROCESS" | awk '{print $4}')
    TIME=$(echo "$PROCESS" | awk '{print $10}')
    echo "✅ Training is RUNNING"
    echo "   PID: $PID"
    echo "   CPU: ${CPU}%"
    echo "   Memory: ${MEM}%"
    echo "   Time: $TIME"
else
    echo "❌ Training is NOT running"
fi

echo ""
echo "=== LATEST PROGRESS ==="
tail -10 training_ultimate.log

echo ""
echo "=== EVALUATION HISTORY ==="
grep "eval:" training_ultimate.log | tail -5

echo ""
echo "=== CHECKPOINTS ==="
ls -lh runs/checkpoints_ultimate/ 2>/dev/null | tail -5

echo ""
echo "=== CURRICULUM TRANSITIONS ==="
grep ">>> Curriculum" training_ultimate.log

echo ""
echo "=== ITERATION COUNT ==="
ITERS=$(grep -c "^iter=" training_ultimate.log)
echo "Completed: $ITERS / 25000 iterations"
PERCENT=$(echo "scale=1; $ITERS * 100 / 25000" | bc)
echo "Progress: ${PERCENT}%"

echo ""
echo "=== COMMANDS ==="
echo "Watch live: tail -f training_ultimate.log"
echo "View TensorBoard: tensorboard --logdir runs/tensorboard_ultimate"
echo "Stop training: ./stop_training.sh"

