# OceanEmbed - Product Requirements Document & Architecture

## 1. Executive Overview
OceanEmbed (MoES/INCOIS PS 26066) is an end-to-end, production-grade application that reconstructs a 3D Subsurface Ocean Temperature Cube across 15 standard depths (0m to 1000m) at 0.25° spatial resolution over the North Indian Ocean (5°N to 30°N, 45°E to 105°E) using daily surface satellite observations.

The system ingests 5 multi-source surface satellite inputs, trains a PyTorch Vision Transformer (ViT) model against GLORYS reanalysis data, validates against independent ARGO float observations, and provides an interactive Streamlit dashboard for real-time inference and disaster diagnostics (MHW, TCHP, ENSO/IOD).

## 2. Directory Layout & File Breakdown

```text
ocean-embed/
├── .env                              # Stores COPERNICUS_USERNAME, COPERNICUS_PASSWORD, EARTHDATA_USERNAME, EARTHDATA_PASSWORD
├── requirements.txt                  # Python dependencies
├── app.py                            # Streamlit frontend application
├── PROJECT_ARCHITECTURE_AND_FILES.md # This documentation file
├── data_pipeline/                    # Data Ingestion & Preprocessing (Module A)
│   ├── download_training.py          # Script for historical bulk download
│   ├── live_fetcher.py               # Real-time data fetcher from PO.DAAC & Copernicus
│   └── harmonizer.py                 # xarray regridding & interpolation logic (0.25° grid)
├── ml/                               # Deep Learning Engine (Module B)
│   ├── model.py                      # OceanEmbedViT & CNN architectures
│   ├── train.py                      # Training loop, loss functions, optimizer
│   └── validate_argo.py              # ARGO float independent validation metrics
├── diagnostics/                      # Physics & Diagnostics (Module C)
│   ├── tchp.py                       # Tropical Cyclone Heat Potential
│   ├── mhw.py                        # Marine Heatwave Detection
│   ├── enso_iod.py                   # ENSO/IOD indices and text generation
│   └── export_nc.py                  # CF-compliant NetCDF-4 exporter
└── models/                           # Saved model weights
```

## 3. How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set up Credentials:**
   Edit the `.env` file and add your credentials:
   ```env
   COPERNICUS_USERNAME=your_username
   COPERNICUS_PASSWORD=your_password
   EARTHDATA_USERNAME=your_username
   EARTHDATA_PASSWORD=your_password
   ```
3. **Run the Data Pipeline (Optional, for training):**
   ```bash
   python data_pipeline/download_training.py
   ```
4. **Train the Model (Optional):**
   ```bash
   python ml/train.py
   ```
5. **Launch the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
