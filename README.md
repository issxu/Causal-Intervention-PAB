# Causal-Intervention-PAB

Code for the paper **"Causal Intervention with LMMs: Generative Text Facilitates Fine-Grained Pedestrian Anomaly Behavior Retrieval."**

This repository provides an X2-VLM-based text-to-image retrieval pipeline for fine-grained pedestrian anomaly behavior search on the PAB benchmark. The training pipeline supports multiple captions and action-/appearance-aware negative captions.

## Repository Structure

```text
.
|-- Search_xvlm2.py          # Main training and evaluation script
|-- run_xvlm2.py             # Single-/multi-GPU launcher
|-- train_xvlm2.py           # Training objectives
|-- eval.py                  # Retrieval evaluation utilities
|-- configs/                 # Dataset and model configurations
|-- dataset/                 # Dataset loaders and augmentation
`-- models/                  # Vision-language model modules
```

Training outputs, Slurm logs, datasets, checkpoints, and pretrained weights are not included in this repository.

## Requirements

We recommend Python 3.10 and a CUDA-enabled PyTorch installation compatible with your system.

```bash
conda create -n pab_x2vlm python=3.10
conda activate pab_x2vlm

# Install a PyTorch version compatible with your CUDA version first.
pip install -r requirements.txt
```

The first time WordNet is used, download its data:

```bash
python -c "import nltk; nltk.download('wordnet')"
```

## Data and Checkpoints

The PAB dataset and pretrained checkpoints must be prepared separately. Edit `configs/pab.yaml` and replace all machine-specific paths with local paths, including:

- `image_root`: PAB image directory;
- `train_file`: training annotation files;
- `test_file`: test annotation file;
- `text_encoder`: local BERT tokenizer/encoder directory.

Training annotations should provide `image`, `image_id`, and caption fields. For the hard-negative training setting, annotations may also provide `negative_action` and `negative_appearance`. Test annotations should provide image identifiers, image paths, and one or more captions.

## Training

From the repository root, run:

```bash
python run_xvlm2.py \
    --task pab \
    --dist auto \
    --output_dir output/pab \
    --checkpoint /path/to/x2vlm_base_1b.th \
    --seed 42
```

Useful overrides include `--bs` (batch size), `--epo` (number of epochs), and `--lr` (learning rate). The corresponding Slurm scripts can be adapted to your cluster.

## Evaluation

Evaluate a trained checkpoint with:

```bash
python run_xvlm2.py \
    --task pab \
    --dist auto \
    --output_dir output/pab_eval \
    --checkpoint output/pab/checkpoint_best.pth \
    --evaluate
```

Other dataset configurations are available in `configs/` (for example, `baseline`, `cuhk`, `icfg`, and `rstp`).

## Citation

```bibtex
@inproceedings{xu2026causal,
  title     = {Causal Intervention with LMMs: Generative Text Facilitates Fine-Grained Pedestrian Anomaly Behavior Retrieval},
  author    = {Xu, Weifeng and Huang, Shaofei and Wang, Yaxiong and Zheng, Zhedong},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

## License

This code is intended for research use. Please check the licenses and usage terms of the PAB data, pretrained models, and third-party components before redistribution.
