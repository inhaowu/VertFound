python train_net.py --num-gpus 8 \
    --config-file configs/Base-VertFound.yaml \
    --dist-url auto \
    TEST.EVAL_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 32 \
    SOLVER.MAX_ITER 10000 \
    MODEL.WEIGHTS weights/vertfound_bl.pth \
    MODEL.CLIP_TYPE CLIP_400M_Large \
    MODEL.CLIP_INPUT_SIZE 224 \
    MODEL.BOX_TYPE 'GT' \
    DATASETS.TRAIN '("your_dataset",)'\
    DATASETS.TEST '("your_dataset",)' \
    SOLVER.CHECKPOINT_PERIOD 100 \
    OUTPUT_DIR ./output/verse19_vertfound