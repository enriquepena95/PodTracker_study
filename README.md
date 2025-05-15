# PodTracker Study

This repository contains code to recreate results reported in Enhancing Peanut Pre-Sizing with PodTracker: A Multiple Pod Tracking Algorithm by Pena Martinez et al. (2025). The system uses Mask R-CNN for detection and two tracking approaches: a simple deterministic IoU and Euclidean-based tracker (PodTracker) and DeepSORT (DeepSortMask).

## Setup Instructions

### 1. Clone the Repository

Clone this repository to your home directory:

```bash
cd ~
git clone https://github.ncsu.edu/eepena/PodTracker_study/
cd PodTracker_study
```

### 2. Clone Required Dependencies

Clone Detectron2 and DeepSortMask into the PodTracker_study directory:

```bash
# Clone Detectron2
git clone https://github.com/facebookresearch/detectron2.git

# Clone DeepSortMask
git clone https://github.com/zafarRehan/DeepSortMask.git
```

### 3. Modify DeepSortMask Files

Several modifications are needed to ensure compatibility with the latest Python and NumPy versions:

#### a. Update the cosine distance function in `nn_matching.py`:

Edit `~/PodTracker_study/DeepSortMask/deep_sort/nn_matching.py` and replace the `_cosine_distance()` function with:

```python
def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to length 1.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that element (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """
    # Ensure a and b are 2D arrays
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Reshape if they are 3D
    if len(a.shape) > 2:
        a = np.reshape(a, (-1, a.shape[-1]))
    if len(b.shape) > 2:
        b = np.reshape(b, (-1, b.shape[-1]))
    
    # Apply normalization only once
    if not data_is_normalized:
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
    
    return 1. - np.dot(a, b.T)
```

#### b. Update NumPy data types:

1. In `~/PodTracker_study/DeepSortMask/deep_sort/detection.py`:
   - Change lines 32-35: replace `dtype=np.float` with `dtype=np.float64`

2. In `~/PodTracker_study/DeepSortMask/deep_sort/preprocessing.py`:
   - Change line 41: replace `dtype=np.float` with `dtype=np.float64`

3. In `~/PodTracker_study/DeepSortMask/toolsTracker/generate_detections.py`:
   - Change line 63: replace `dtype=np.float` with `dtype=np.float64`

### 4. Create and Activate Conda Environment

Create a conda environment using the provided YAML file:

```bash
conda env create -f ~/PodTracker_study/ml_env.yml
conda activate ml_env
```

### 5. Download and Extract Data

1. Download the dataset from Dryad: [[DOI: 10.5061/dryad.v41ns1s7p](https://doi.org/10.5061/dryad.v41ns1s7p)](http://datadryad.org/share/WteqhMqjfH4gURPfTDwTjmleMMs9-kJMabYQjO13hiU)

2. Extract the 7zip archive to the PodTracker_study directory:
   ```bash
   cd ~/PodTracker_study
   7z x data.7z.001
   ```
   
   Note: This requires the 7-Zip utility. If you don't have it installed, you can install it with:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install p7zip-full
   
   # CentOS/RHEL
   sudo yum install p7zip p7zip-plugins
   
   # macOS (using Homebrew)
   brew install p7zip
   ```

### 6. Run the Analysis

Make the run script executable and run it:

```bash
chmod +x ~/PodTracker_study/run-analysis.sh
./run-analysis.sh
```

## Expected Project Structure After Execution

```
PodTracker_study/
├── code/                     # Python scripts for pod analysis
├── data/                     # Dataset (after extraction)
│   ├── grade_data/           # Grading-related data
│   └── counts_data/          # Counting-related data
│   ├── decision_tree_data/   # Grading-related data
│   └── pod_model_data/       # Counting-related data
├── detectron2/               # Detectron2 library (cloned)
├── DeepSortMask/             # DeepSortMask library (cloned and modified)
├── model_checkpoints/        # Trained model weights
│   └── sp_model.pth          # Initial model used for training a new peanut pod model 
├── ml_env.yml                # Conda environment file
├── results/                  # Output results and visualizations
│   ├── plots/                # Generated plots
│   └── tables/               # Generated data tables
└── run-analysis.sh           # Main execution script
└── log.txt                   # Model training log file
```

## Contact

For questions or support, please contact [enriquepena1995@gmail.com]
