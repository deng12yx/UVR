# UVR: Unified Visual Safety Regulation for Multimodal Diffusion Transformers

[![Conference](https://img.shields.io/badge/ICML-2026-blue)](#)
[![Task](https://img.shields.io/badge/Task-T2I%20%7C%20I2I-green)](#)
[![Method](https://img.shields.io/badge/Method-Training--Free-orange)](#)

Official implementation of **UVR**, a training-free safety regulation framework for multimodal diffusion transformers. This work has been accepted to **ICML 2026**.

UVR targets safety-critical image generation and editing under a unified generation-time regulation framework. Instead of retraining model weights or filtering outputs after generation, UVR localizes unsafe visual information during inference and restricts its propagation inside the multimodal generation process. The same mechanism supports both **text-to-image generation** and **image-to-image editing**.

> **Paper status.** The official paper link, project page, and BibTeX will be updated once the ICML 2026 proceedings or public preprint are available.

## News

- **2026.05**: UVR has been accepted to **ICML 2026**.
- Code is being organized for public release. The current version provides the core scripts for unsafe anchor collection, T2I safety regulation, and I2I safety regulation.

## Method Overview

UVR follows an **offline-online** design:

1. **Offline unsafe anchor collection**  
   UVR first collects unsafe visual anchors from unsafe examples. These anchors are extracted from the image-token/output embedding space of the multimodal diffusion transformer and serve as reusable references for unsafe visual concepts.

2. **Online unsafe region localization**  
   During generation or editing, UVR compares intermediate visual features with the collected anchors to localize regions that are likely to carry unsafe information.

3. **Generation-time safety regulation**  
   Once unsafe regions are localized, UVR intervenes during inference to restrict unsafe information flow, while preserving the overall generation quality and semantic alignment as much as possible.

This design makes UVR lightweight and model-adaptive: the unsafe anchor set is constructed once, while safety regulation is performed only at inference time.

## Highlights

- **Accepted to ICML 2026**
- **Training-free**: no fine-tuning or model-weight modification is required
- **Unified T2I and I2I safety control** under the same MM-DiT/MM-Attention framework
- **Offline-online design**: reusable unsafe anchors with online localization and intervention
- **Inference-time regulation** of unsafe visual information flow
- Built on top of Hugging Face `diffusers` and FLUX-family pipelines

## Repository Structure

```text
.
├── config.py                         # Model paths and UVR hyperparameters
├── unsafe_anchor_collection.py        # Offline unsafe anchor construction
├── UVR_T2I.py                         # Text-to-image safety-regulated generation demo
├── UVR_I2I.py                         # Image-to-image safety-regulated editing demo
├── UVR_T2I.ipynb                      # T2I visualization notebook
├── UVR_I2I.ipynb                      # I2I visualization notebook
└── my_flux/                           # Customized FLUX pipeline and transformer modules
```

## Installation

Create a Python environment and install the main dependencies:

```bash
conda create -n uvr python=3.10 -y
conda activate uvr

pip install torch torchvision
pip install diffusers transformers accelerate pillow scipy numpy
```

The implementation is designed for FLUX-family models. By default, the configuration uses:

- `black-forest-labs/FLUX.1-dev` for text-to-image generation
- `black-forest-labs/FLUX.1-Kontext-dev` for image-to-image editing

Please make sure that you have accepted the corresponding model licenses on Hugging Face and have access to the checkpoints.

You can update the model paths in `config.py`:

```python
Dev_model_path = "black-forest-labs/FLUX.1-dev"
Kontext_model_path = "black-forest-labs/FLUX.1-Kontext-dev"
```

## Usage

### 1. Collect Unsafe Anchors Offline

Run the offline anchor construction script:

```bash
python unsafe_anchor_collection.py
```

This step collects unsafe anchor embeddings from the model output space. The collected anchors can then be reused for both T2I generation and I2I editing under the same safety regulation pipeline.

### 2. Run Text-to-Image Safety Regulation

Edit the prompt and generation settings in `UVR_T2I.py`, then run:

```bash
python UVR_T2I.py
```

The script loads FLUX.1-dev, applies the customized UVR modules, and performs safety-regulated text-to-image generation.

### 3. Run Image-to-Image Safety Regulation

Place the source image at:

```text
unsafe.jpg
```

Edit the instruction prompt and generation settings in `UVR_I2I.py`, then run:

```bash
python UVR_I2I.py
```

The edited output is saved as:

```text
flux.jpg
```

### 4. Visualization Notebooks

The notebooks provide qualitative examples and intermediate visualizations:

- `UVR_T2I.ipynb`
- `UVR_I2I.ipynb`

## Configuration

Important parameters are defined in `config.py`:

```python
threshold = 0.6   # Safety localization threshold for FLUX.1-dev
last_step = 850   # Timestep threshold for intervention
pre_thr = 0.2     # Pre-threshold for FLUX.1-Kontext-dev
```

These values can be adjusted for different safety concepts, models, or localization-regulation trade-offs.

## Practical Notes

- The unsafe anchor collection stage only needs to be run once for a given setup.
- UVR performs safety regulation at inference time and does not modify model weights.
- The demo scripts are intentionally minimal so that the core mechanism is easy to inspect and adapt.
- Local checkpoints, cached model files, generated images, and collected anchor outputs should not be committed to the repository.
- For different unsafe concepts, the localization threshold may need to be adjusted according to the concept-specific attention/feature distribution.

## Citation

UVR has been accepted to **ICML 2026**. Since the official proceedings version is not public yet, we currently recommend citing this repository only in informal contexts and updating the citation once the paper link or official BibTeX is released.

```text
Citation information will be updated after the official ICML 2026 paper/preprint is available.
```

## License

The source code license will be clarified before the final public release. The underlying FLUX checkpoints are governed by their respective model licenses.

## Acknowledgements

This implementation builds on the open-source ecosystem around PyTorch, Hugging Face `diffusers`, and FLUX-family models. We thank the community for providing the tools that make reproducible research possible.
