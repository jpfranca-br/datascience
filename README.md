# Data Science for Digital Transformation Project

This project was developed for the *Data Science for Digital Transformation* course at PUC-Rio, taught by Professor Dr. Dan Reznik. It performs data analysis, profiling, and predictive modeling on socio-economic and transport data from Rio de Janeiro, Brazil.

## Project Overview

The project integrates and analyzes data on metro ridership, population, and GDP in Rio de Janeiro, aiming to derive insights into socio-economic patterns and predict future trends using neural networks.

### Data Sources

The data sources used in this project include:
1. **Metro Ridership Data (1998-2023)** - Available on Kaggle [here](https://www.kaggle.com/datasets/lxncylot/brazil-rio-de-janeiro-subway-1998-2023).
2. **GDP Data for Rio de Janeiro (1999-2021)** - Retrieved from IBGE [here](https://cidades.ibge.gov.br/brasil/rj/rio-de-janeiro/pesquisa/38/47001?indicador=46997&ano=2021).
3. **Population Data for Rio de Janeiro (1970, 1980, 1990-2024)** - From Data Rio [here](https://www.data.rio/documents/90106eb8874f4e8fbbc27678bbb1e772/about).

## Features

1. **Data Cleaning**:
   - Cleans and preprocesses the metro ridership data, handling missing values and data formatting issues.
   - Interpolates missing values in population data.
   - Processes GDP data to unify multiple GDP indicators into a single series.

2. **Data Profiling and Analysis**:
   - Generates data profiles including basic statistics and unique value analysis.
   - Identifies missing values and their distribution across datasets.
   - Conducts exploratory data analysis using visualizations (scatter plots, line charts, box plots, and heatmaps).

3. **Predictive Modeling**:
   - Trains a neural network model to forecast future ridership, population, and GDP trends.
   - Uses a min-max scaling approach for normalization.
   - Visualizes actual vs. predicted values.

## Installation

To use this code, you need Python 3.11 and the following libraries:

```bash
pip install pandas matplotlib scikit-learn seaborn tensorflow
```

## Usage

1. **Configuration**:
   - Edit `config.py` to adjust settings for plotting, data prediction, and debugging. Parameters like `split_year`, `epochs`, and `batch_size` can be modified to control the training behavior.

2. **Running the Code**:
   - Run `main.py` to execute the full analysis. This will:
     - Profile and clean the metro, population, and GDP data.
     - Perform exploratory data analysis.
     - Train a neural network model if `PREDICTION` is enabled in `config.py`.

3. **Output**:
   - Plots and prediction results are saved in the `images` directory if `SAVE` is enabled in `config.py`.

## File Structure

- **`main.py`**: Main script for data loading, profiling, analysis, and prediction.
- **`config.py`**: Configuration file for adjustable settings.
- **`user_data_lib.py`**: Library for data profiling and prediction functions.
- **`user_graph_lib.py`**: Library for visualization functions.
- **`user_text_lib.py`**: Library for text and debug output functions.

--- 

