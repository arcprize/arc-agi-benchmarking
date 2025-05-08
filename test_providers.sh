#!/bin/bash

set -e  # Exit on any error

# Run test for a provider
run_test() {
    local config_name=$1
    local task_id=$2
    
    
    echo "🔄 Testing task: ${task_id}, config: ${config_name:-default}"
    
    # Build the command with model as the config
    local cmd="python3 -m main --data_dir data/arc-agi/data/evaluation --config \"${config_name}\" --task_id \"${task_id}\" --save_submission_dir ."
    
    # Add config_name if provided (for future use if needed)
    # if [ -n "$config_name" ]; then
    #     cmd="$cmd --config_name \"${config_name}\""
    # fi
    
    # Execute the command
    if eval $cmd; then
        echo "✅ Test completed successfully for ${config_name:+/}${config_name}"
        return 0
    else
        echo "❌ Test failed for ${config_name:+/}${config_name}"
        return 1
    fi
}

export -f run_test

echo "Starting provider tests..."

# Define test configurations
CONFIGS=(
    "gpt-4-5-preview-2025-02-27 14754a24"
    "gpt-4-1-2025-04-14 7bb29440"
    "gpt-4o-2024-11-20 f0afb749"
    "claude-3-7-sonnet-20250219 dc2e9a9d"
    "claude_opus f83cb3f6"
    "gemini-1-5-pro-002 baf41dbf"
    "deepseek_chat 93b4f4b3"
    "o3-mini-2025-01-31-high 136b0064"
    # "QwQ-32B d4b1c2b1"
    "QwQ-32B-Fireworks e57337a4"
    "grok-3-mini-beta 0934a4d8"
    "grok-3-beta d4b1c2b1"
)

# Create temporary file with active configurations
TMP_CONFIG_FILE=$(mktemp)
for config in "${CONFIGS[@]}"; do
    echo "$config" >> "$TMP_CONFIG_FILE"
done

# Run tests in parallel
if cat "$TMP_CONFIG_FILE" | parallel --halt now,fail=1 --colsep ' ' -j4 "run_test {1} {2}"; then
    echo "✨ All tests completed successfully"
    rm "$TMP_CONFIG_FILE"
    exit 0
else
    echo "💥 Some tests failed"
    rm "$TMP_CONFIG_FILE"
    exit 1
fi