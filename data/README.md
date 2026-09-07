# Data

Neither the training/evaluation datasets nor the EoMT model checkpoints are included in this repository (licensing and size). Set an environment variable pointing to wherever you keep them:

```bash
export MASKARCH_DATA_ROOT=/path/to/your/data
```

All scripts and notebooks default to `./data` (relative to wherever you run them from) if this variable is not set. Expected layout under `$MASKARCH_DATA_ROOT`:

```
$MASKARCH_DATA_ROOT/
├── checkpoints/
│   ├── cityscapes/eomt_cityscapes.bin
│   ├── coco/eomt_coco.bin
│   └── finetuned_v2/phase3_unfreeze_more/epoch=2-step=2232.ckpt
└── datasets/
    ├── cityscapes/                       (leftImg8bit + gtFine)
    └── anomaly/
        ├── RoadAnomaly21/images/*.png     (SMIYC RA-21)
        ├── RoadObstacle21/images/*.webp   (SMIYC RO-21)
        ├── LostAndFound/images/*.png      (Fishyscapes L&F)
        ├── fs_static/images/*.jpg         (Fishyscapes Static)
        └── RoadAnomaly/images/*.jpg       (original Road Anomaly dataset)
```

## Where to get each dataset

- **Cityscapes** — register and download `leftImg8bit` + `gtFine` from [cityscapes-dataset.com](https://www.cityscapes-dataset.com/). Convert `_labelIds` to `_labelTrainIds` with the [official scripts](https://github.com/mcordts/cityscapesScripts).
- **SMIYC (RoadAnomaly21 / RoadObstacle21)** — [segmentmeifyoucan.com](https://segmentmeifyoucan.com/).
- **Fishyscapes (Lost & Found, Static)** — [fishyscapes.com](https://fishyscapes.com/).
- **Road Anomaly (original)** — from Lis et al., *"Detecting the Unexpected via Image Resynthesis"* (ICCV 2019).

## EoMT checkpoints

- COCO-pretrained and Cityscapes-trained checkpoints: see the [official EoMT repo](https://github.com/tue-mps/eomt) for released weights, or train your own with `eomt/main.py` and the configs in `eomt/configs/`.
- The fine-tuned (LLRD) checkpoint is this project's own output — retrain it with `eomt/finetuning.ipynb` using the Cityscapes-pretrained checkpoint as a starting point.

## ERFNet checkpoints

Already included in [`trained_models/`](../trained_models) (small enough to version directly): `erfnet_pretrained.pth` and `erfnet_encoder_pretrained.pth.tar`.
