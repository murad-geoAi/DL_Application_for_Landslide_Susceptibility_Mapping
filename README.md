# Deep Learning Application for Landslide Susceptibility Mapping

## Overview
This research project applies Deep Learning techniques, specifically Convolutional Neural Networks (CNN), to map landslide susceptibility. The goal is to address spatial sparsity in data and provide accurate susceptibility maps to aid in disaster management and planning.

## Key Features
- **Deep Learning Model**: Utilizes CNNs to process spatial data for landslide prediction.
- **Data Preprocessing**: Handles missing values using KNN and Simple Imputers.
- **Visualization**: Generates susceptibility maps and analyzes feature distributions.

## Directory Structure
The repository is organized as follows:
```
├── data/
│   ├── raw/            # Original data (e.g., Labeled Data Set.csv)
│   └── processed/      # Cleaned and processed data
├── docs/               # Project documentation and presentations
├── notebooks/          # Jupyter notebooks for analysis and modeling
├── reports/            # Generated figures and reports
│   └── figures/
├── src/                # Source code (scripts, utilities)
├── README.md           # Project overview and instructions
└── requirements.txt    # Python dependencies
```

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Dependencies listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/murad-geoAi/DL_Application_for_Landslide_Susceptibility_Mapping.git
   ```
2. Navigate to the project directory:
   ```bash
   cd DL_Application_for_Landslide_Susceptibility_Mapping
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Data Setup
Place your dataset (e.g., `Labeled Data Set.csv`) in the `data/raw/` directory. The notebook expects the file at `../data/raw/Labeled Data Set.csv`.

### Usage
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `notebooks/landslide_susceptibility_mapping.ipynb`.
3. Run the cells to perform data cleaning, analysis, and model training.

## Results
### Study Area
![Study Area Map](reports/figures/study_area_map.jpg)

### Landslide Susceptibility Map (CNN)
![LSM by CNN](reports/figures/lsm_by_cnn.png)

## License
[MIT License](LICENSE) (or appropriate license)
