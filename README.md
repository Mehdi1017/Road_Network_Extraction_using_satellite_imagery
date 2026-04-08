# Road Network Detection and Route Travel Time Estimation from Satellite Imagery

This repository contains the implementation of my Master's Thesis for the **Erasmus Mundus Master in Geospatial Technologies**. The project addresses the "Topology Gap" in automated road extraction by contrasting hierarchical Transformers against optimized CNN architectures.

## Key Highlights

* **Architectural Comparison:** Contrast between **SegFormer (MiT-B3)** with Self-Attention and **DeepLabV3+** with a custom **D3S2PP** module.
* **Topological Repair:** Integrated geometric post-processing heuristics (Filin et al. & Li et al.) to ensure network connectivity.
* **Advanced Evaluation:** Evaluation framework utilizing **APLS (Average Path Length Similarity)** and **Weisfeiler-Lehman (WL) Subtree Kernels** to measure structural isomorphism.

## Methodology

The pipeline consists of four main stages:

1. **Data Preparation:** Radiometric normalization of 11-bit SpaceNet imagery and buffer rasterization.
2. **Feature Extraction:** Hierarchical encoding using Self-Attention (global context) vs. Atrous Convolutions (local context).
3. **Graph Extraction:** Skeletonization and vectorization of probability masks.
4. **Network Evaluation:** Functional routing analysis beyond standard pixel-wise metrics (IoU).

## Repository Structure

```text
├── models/             # SegFormer (MiT-B3) and DeepLabV3+ w/ D3S2PP
├── post_processing/    # Heuristics for topology repair (Filin, Li)
├── utils/              # Metrics: APLS, WL-Kernel, and IoU calculations
├── weights/            # (Download links in README)
├── samples/            # Sample imagery for quick-start inference
├── main.py             # Training entry point
└── predict.py          # Inference and vectorization script
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Mehdi1017/Road_Network_Extraction_using_satellite_imagery.git
cd Road_Network_Extraction_using_satellite_imagery

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run inference on a sample image
python predict.py --input samples/mumbai_sample.tif --model mit_b3
```
## Usage Guide

This repository uses modular entry-point scripts. You can see all available arguments for any script by passing the `--help` flag (e.g., `python train.py --help`).

### 1. Data Preprocessing
Convert the raw SpaceNet GeoJSON labels into 2-meter buffered raster masks. This script automatically processes all city directories found in `data/raw/`.

```bash
python src/data_prep/preprocess.py
```

### 2. Data Splitting
Generate the training, validation, and testing lists. The script writes relative paths to `data/splits/`.

```bash
# Generate both Combined and Per-City splits (Default)
python src/data_prep/split_data.py 

# Generate ONLY the combined cross-city splits
python src/data_prep/split_data.py --mode combined --ratio 0.8
```

### 3. Model Training
Train the network architectures. The script automatically handles loss function switching, learning rate scheduling, and early stopping.

```bash
# Baseline: Train ResNet50 with standard Pixel Loss (BCE/Dice)
python train.py --model resnet50 --loss pixel --batch_size 4 --epochs 100

# Thesis Contribution: Train custom D3S2PP with Topology-Aware Loss
python train.py --model d3s2pp --loss topo --epochs 100 --warmup 5

# Transformer: Train SegFormer MiT-B3 
python train.py --model mit_b3 --loss topo --batch_size 4

# Resume a crashed or stopped training run
python train.py --model mit_b3 --loss topo --resume
```
### 4. Inference & Post-Processing
The `predict.py` script handles model inference and applies specific topological repair heuristics (Filin et al., Li et al.) to the predicted masks. It can process a single image or an entire directory.

**Basic Usage:**
```bash
python predict.py --test_list data/splits/test_list_AOI_5_Khartoum_full.txt \
                  --model d3s2pp \
                  --weights weights/d3s2pp_resnet50_best.pth \
                  --post_process li \
                  --threshold 0.5
```
## Results (Vegas)

| Model | IoU | APLS | WL-Kernel | APLS_Time |
| :--- | :---: | :---: | :---: | :---: |
| ResNet50 + D3S2PP | 0.45 | 0.64 | 0.82 | 0.46 |
| **MiT-B3 + Unet** | **0.35** | **0.78** | **0.85** | **0.40** |

## Citation

```bibtex
@mastersthesis{gassa2026road,
  author  = {Gassa Malki, El Mehdi},
  title   = {Road Network Detection and Route Travel Time Estimation from Satellite Imagery},
  school  = {Erasmus Mundus Master in Geospatial Technologies},
  year    = {2026}
}
```

## License

Distributed under the MIT License.

## Contact & Acknowledgements

**El Mehdi Gassa Malki** - [LinkedIn](https://www.linkedin.com/in/el-mehdi-gassa-malki) - mehdigassamalki@gmail.com

Special thanks to my thesis supervisors Filipe Feitosa, Marco Painho, and Marcia Baptista, as well as the Erasmus Mundus GeoTech program for their support.