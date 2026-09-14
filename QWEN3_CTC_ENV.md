Qwen3 CTC tone contrastive 环境使用 Python 3.11，独立安装在项目的 `.venv-qwen3-ctc` 中。

重新创建环境：

```bash
BASE_PYTHON=/home/xintong/miniconda3/bin/python bash setup_qwen3_ctc_env.sh
```

启动四卡训练：

```bash
bash run_jsonl_qwen3_ctc_tonecontrast_4gpu.sh
```

Bucket 声学对比版本（同韵母组 batch，无可学习音素 prototype）：

```bash
bash run_jsonl_qwen3_ctc_finalbucket_4gpu.sh
bash resume_jsonl_qwen3_ctc_finalbucket_4gpu.sh
bash run_qwen3_bucket_tone_tsne_i_500.sh
```

默认冻结 backbone 前 10k 次 optimizer 更新；对比损失在 10k 后启用，
2k 次更新内将权重升至 0.05。每卡 microbatch 为 4，4 卡累积 3 次梯度，
总更新上限为 30k。Bucket epoch 遍历完整 sampler。

Resume 默认加载实验目录下的 `checkpoints/last.ckpt`；可通过
`RESUME_CHECKPOINT` 指定文件。恢复模型、optimizer、scheduler 和混合精度状态。
Ctrl+C 或 SIGTERM 会等待当前梯度累积完成，在退出前保存完整 `last.ckpt`；
正常训练结束也保存最终状态。SIGKILL、断电无法触发此保存。
当前 DataLoader 不保证在 epoch 中途恢复时精确续接样本顺序。

t-SNE 脚本使用真实音频 `l2_wav` 和 `actual_phones`，提取 i1--i5 各 500 个
CTC 对齐特征，在 50 维 PCA 空间计算 Silhouette Score，并输出 PNG/PDF。
`CHECKPOINT` 和 `OUTPUT_DIR` 可覆盖默认路径。

启动脚本自动定位项目目录和独立 Python，无需激活 Conda。`PYTHON_BIN` 可覆盖默认解释器。
模型缓存默认位于 `.cache-qwen3/huggingface`，可用 `HF_HOME` 覆盖。
`DATA_ROOT`、`SPLIT_DIR`、`EXP_DIR`、`CUDA_VISIBLE_DEVICES` 和对比学习环境变量均可覆盖。

该环境专用于 Qwen3，不使用根目录旧版 Whisper 的 requirements.txt。
原生 Qwen3-ASR 需要 Transformers >=5.13.0：
https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf
