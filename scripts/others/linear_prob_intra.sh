# timestamp=$(date "+%Y%m%d_%H%M%S")
# echo ${timestamp}

# expname1=exp/exp_flir/flir_lora_MVQ1024x8_256x8

# python application/linear_prob.py \
# --expname ${expname1} \
# --prob_modality vision thermal \
# --embed_type common specific \
# --device cuda:0 \
# --logger | tee ${expname1}/${timestamp}_train_lp.log

# 1. 生成时间戳
timestamp=$(date "+%Y%m%d_%H%M%S")
echo "Training started at: ${timestamp}"

# 2. 设置参数 (在这里修改 swap 状态)
SWAP=false  # 或者 false

# 3. 构造与 Python 逻辑一致的 save_name
if [ "$SWAP" = true ]; then
    swap_status="true"
else
    swap_status="false"
fi

# 这里的路径结构要和 Python main 函数中的 os.path.join("./exp", save_name, "log_lp") 匹配
SAVE_NAME="exp_intra_swap_${swap_status}_lp"
LOG_DIR="./exp/${SAVE_NAME}/log_lp"

# 4. 显式创建日志目录，防止 tee 报错
mkdir -p ${LOG_DIR}

# 5. 执行 Python 脚本
# 注意：--swap 是 action="store_true"，所以如果是 true 就加上这个 flag
SWAP_FLAG=""
if [ "$SWAP" = true ]; then
    SWAP_FLAG="--swap"
fi

python application/linear_prob_intra.py \
    --max_epochs 20 \
    --batch_size 100 \
    --device cuda:0 \
    ${SWAP_FLAG} \
    --logger \
    2>&1 | tee ${LOG_DIR}/${timestamp}_train_lp.log
