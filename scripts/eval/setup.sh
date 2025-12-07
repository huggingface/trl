# ==============================================================================
# 函数：run_single
# 描述：执行单次评测
# 参数：
#   $1: cmd_prefix (torchrun 的前缀命令)
#   $2: data_names (数据集名称，如 "math", "gsm8k")
#   $3: model_path (具体的模型/Checkpoint 路径)
#   $4: output_dir (输出目录)
# ==============================================================================
function run_single() {
    local cmd_prefix="$1"
    local model_path="$2"
    local output_dir="$3"

    local full_cmd="${cmd_prefix} --model_name_or_path ${model_path} --output_dir ${output_dir}"
    if [ -n "$SLURM_PROCID" ] && [ "$SLURM_PROCID" -eq 0 ]; then
        echo ">>> Full Command: ${full_cmd}"
    fi
    $full_cmd | grep -v -E "\[Gloo\]|EngineCore|connected peer ranks|PYTORCH_CUDA_ALLOC_CONF|Loading safetensors"
}

# ==============================================================================
# 函数：run_loop
# 描述：循环遍历 checkpoint 进行评测
# 参数：
#   $1: cmd_prefix      (torchrun 的前缀命令)
#   $2: data_names      (数据集名称)
#   $3: base_model_path (包含 checkpoint-xxx 的父目录)
#   $4: base_output_dir (输出父目录)
#   $5: start_step      (起始步数)
#   $6: interval_step   (步长)
#   $7: end_step        (结束步数)
# ==============================================================================
function run_loop() {
    local cmd_prefix="$1"
    local base_model_path="$2"
    local base_output_dir="$3"
    local start_step="$4"
    local interval_step="$5"
    local end_step="$6"

    if [ -n "$SLURM_PROCID" ] && [ "$SLURM_PROCID" -eq 0 ]; then
        echo "=================================================="
        echo "Starting Loop Eval for dataset: ${data_names}"
        echo "Model Base: ${base_model_path}"
        echo "Steps: ${start_step} to ${end_step} (Interval: ${interval_step})"
        echo "=================================================="
    fi

    for step in $(seq $start_step $interval_step $end_step); do
        local ckpt_path="${base_model_path}/checkpoint-${step}"
        local ckpt_output="${base_output_dir}/checkpoint-${step}"

        if [ -n "$SLURM_PROCID" ] && [ "$SLURM_PROCID" -eq 0 ]; then
            echo ">>> Full Command: ${full_cmd}"
        fi     
   
        # 调用 run_single 执行
        run_single "$cmd_prefix" "$ckpt_path" "$ckpt_output"
        
        if [ -n "$SLURM_PROCID" ] && [ "$SLURM_PROCID" -eq 0 ]; then
            echo ">>> [Loop Progress] Step ${step} Finished."
            echo ""
        fi
    done
}