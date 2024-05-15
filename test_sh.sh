python train_net.py --num-gpus 8 \
    --config-file configs/eval.yaml \
    --eval-only True \
    --dist-url auto \
    MODEL.WEIGHTS /path/to/your/weight \
    MODEL.CLIP_TYPE CLIP_400M_Large \
    MODEL.CLIP_INPUT_SIZE 224 \
    MODEL.BOX_TYPE 'GT' \
    DATASETS.TRAIN '("your_dataset",)'\
    DATASETS.TEST '("your_dataset",)'\
    SOLVER.IMS_PER_BATCH 32 \
    # SOLVER.CHECKPOINT_PERIOD 100 \
    # CUDA_LAUNCH_BLOCKING=1 \
    # MODEL.DEVICE cpu # Train with cpu