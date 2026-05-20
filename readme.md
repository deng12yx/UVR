# UVR

Official code release for **UVR**, accepted to **ICML 2026**.

UVR is a training-free safety regulation framework for FLUX-based image generation and editing. The pipeline follows the offline-online design described in the paper: unsafe visual anchors are collected once in an offline stage, and then reused at inference time for online localization and intervention. No model retraining is required.

> Paper, project page, and citation information will be updated after the ICML 2026 proceedings are public.

## Highlights

- **Accepted by ICML 2026**
- **Training-free safety regulation** for diffusion/flow-based generation
- **Offline unsafe anchor collection** followed by **online intervention**
- Supports both **Text-to-Image** generation and **Image-to-Image** editing
- Built on top of Hugging Face `diffusers` and FLUX-family pipelines

## Repository Structure

```text
.
|-- config.py                         # Model paths and UVR hyperparameters
|-- unsafe_anchor_collection.py        # Offline unsafe anchor collection
|-- UVR_T2I.py                         # Text-to-Image demo
|-- UVR_I2I.py                         # Image-to-Image demo
|-- UVR_T2I.ipynb                      # T2I visualization notebook
|-- UVR_I2I.ipynb                      # I2I visualization notebook
`-- my_flux/                           # Customized FLUX pipeline/transformer modules
```

## Environment

Create a Python environment with PyTorch and the main dependencies:

```bash
pip install torch torchvision diffusers transformers accelerate pillow scipy numpy
```

The default configuration uses:

- `black-forest-labs/FLUX.1-dev`
- `black-forest-labs/FLUX.1-Kontext-dev`

Please make sure you have accepted the corresponding model licenses on Hugging Face and have access to the checkpoints. If needed, update the model paths in `config.py`:

```python
Dev_model_path = "black-forest-labs/FLUX.1-dev"
Kontext_model_path = "black-forest-labs/FLUX.1-Kontext-dev"
```

## Usage

### 1. Offline Unsafe Anchor Collection

Run the offline anchor construction once:

```bash
python unsafe_anchor_collection.py
```

This step collects unsafe anchor embeddings from the model output space. The anchors are then reused during both generation and editing.

### 2. Text-to-Image Safety Regulation

Edit the prompt in `UVR_T2I.py`, then run:

```bash
python UVR_T2I.py
```

The script loads FLUX.1-dev, patches the customized UVR modules, and performs safety-regulated text-to-image generation with `perturb_flag=True`.

### 3. Image-to-Image Safety Regulation

Place the source image at:

```text
unsafe.jpg
```

Edit the prompt in `UVR_I2I.py`, then run:

```bash
python UVR_I2I.py
```

The edited output is saved as:

```text
flux.jpg
```

### 4. Visualization

The notebooks provide qualitative examples and intermediate visualizations:

- `UVR_T2I.ipynb`
- `UVR_I2I.ipynb`

## Configuration

Key parameters are defined in `config.py`:

```python
threshold = 0.6   # Safety localization threshold for FLUX.1-dev
last_step = 850   # Timestep threshold for intervention
pre_thr = 0.2     # Pre-threshold for FLUX.1-Kontext-dev
```

You can tune these values for different safety/localization trade-offs.

## Notes

- The unsafe anchor collection stage only needs to be run once for a given setup.
- UVR performs safety regulation at inference time and does not modify model weights.
- The demo scripts are intentionally minimal so that the core method is easy to inspect and adapt.
- Generated images, local checkpoints, cached model files, and anchor outputs should not be committed to the repository.

## Citation

If you find this project useful, please cite our ICML 2026 paper. The BibTeX entry will be added once the official proceedings version is available.

```bibtex
@inproceedings{uvr2026,
  title     = {UVR},
  author    = {To be updated},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

## License

The source code license will be clarified before the final public release. The underlying FLUX checkpoints are governed by their respective model licenses.
