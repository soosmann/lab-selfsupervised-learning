# Self-Supervised Learning for Image Classification

A deep learning lab exploring whether learning to predict image rotations helps a neural network learn useful features for image classification. Implemented in **PyTorch**, the project compares rotation-based self-supervised pretraining (RotNet) with supervised training from scratch on **miniImageNet**.

## Approach

1. **Pretrain:** Train a CNN to recognize rotations of 0°, 90°, 180°, and 270°, using automatically generated labels.
2. **Transfer:** Transfer the learned weights to an image classifier and fine-tune on labeled images.
3. **Compare:** Train the same architecture from scratch and compare accuracy, loss curves, and t-SNE visualizations.

The notebook includes a compact ResNet architecture, an optional Inception-style block, configurable data augmentation, and training and validation routines. Images are loaded at 84 × 84 pixels and cropped to 64 × 64 for training.

## Repository Contents

- [Notebook](src/Hofmann_Celaya_DLISP_Lab2_notebook.ipynb): implementation, training workflow, and visualizations.
- [Lab report](docs/Hofmann_Celaya_DLISP_Lab2_report.pdf): experiment analysis and discussion.

## Getting Started

Open the notebook in Jupyter or Google Colab. A CUDA-capable GPU is recommended for training.

For a local environment, install the dependencies:

```bash
pip install torch torchvision numpy matplotlib pillow scikit-learn wget jupyter
```

Run the notebook cells in order. The dataset download cell retrieves and extracts miniImageNet from the course server. Adjust the `configuration` class to select the architecture, dataset size, training epochs, and augmentation settings.

## Acknowledgments

Based on the rotation-based self-supervised learning lab by Marcos Escudero-Viñolo, with material adapted from Diego Ortego, as credited in the notebook.
