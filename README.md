# Reverse Engineering the op.gg Algorithm 

## Table of Contents
- [Project Overview](#project-overview)
- [Data Scraping](#data-scraping)
- [Data Cleaning](#data-cleaning)
- [Finalized Data Schema](#finalized-data-schema)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Initial Model Creation and Evaluation](#initial-model-creation-and-evaluation)
- [Neural Network Implementation for OP Score Prediction](#neural-network-implementation-for-op-score-prediction-in-league-of-legends)
- [Testing and Evaluating the Neural Network Models](#testing-and-evaluating-the-neural-network-models)
- [Dependencies](#dependencies)
- [Credits](#credits)

## Project Overview

This project aims to reverse engineer the scoring algorithm used by op.gg, a popular platform for League of Legends player statistics and game analysis. The goal is to understand and replicate the evaluation metrics that generate a player's score based on in-game performance. This repository includes all necessary components such as data scraping, preprocessing, model development, and evaluation notebooks.

## Data Scraping

The project includes a set of Python scripts that automate the extraction of game data from op.gg. These scripts utilize `Selenium` for browser automation and `BeautifulSoup` for parsing HTML content to extract detailed game statistics and player performances. The data collection process is structured as follows:

1. **Web Driver Setup**: Configures and initializes a Selenium WebDriver to interact with web pages.
2. **Navigating to URLs**: The script navigates to specific op.gg URLs where data is dynamically loaded, making it available for extraction.
3. **HTML Source Extraction**: Once the page is fully loaded, the HTML source of the page is extracted.
4. **JSON Data Extraction**: The script searches through the HTML to find embedded JSON data that contains detailed information about the games and players.
5. **Data Parsing**: Extracted data is then parsed to focus on key statistics needed for analysis. The parsing function isolates specific elements like player scores, game outcomes, and detailed in-game statistics.
6. **Data Storage**: After extraction and parsing, data is formatted into a pandas DataFrame and stored in CSV format for further processing.

The script handles potential errors and ensures that the data is consistently captured, making it robust for large-scale data scraping operations. This data serves as the foundation for building and training the machine learning models to reverse engineer the op.gg scoring algorithm.

## Data Cleaning

The data cleaning process refines the raw data extracted from op.gg to ensure it's suitable for model training and analysis. This process involves several key steps to enhance the data quality and relevance:

1. **Removing Duplicates**: Eliminates any duplicate records to prevent biased analysis.
2. **Data Transformation**: Includes converting game lengths from seconds to minutes, mapping champion IDs to names, and converting game results into binary format (win/lose).
3. **Metric Calculation**:
   - **Combined Metrics**: Calculates total and percentage metrics such as total kills, gold percentage, and damage percentage for groups representing teams in the games.
   - **Differential Metrics**: Computes the differences in creep score, gold earned, level, and damages between opponents in the same game and position, which is crucial for comparative analysis.
   - **Other Metrics**: Calculates additional performance metrics like KDA, damage per gold, and vision scores.
4. **Data Enrichment**: Incorporates external data, such as champion names, to make the dataset more interpretable.
5. **Feature Selection**: Retains relevant features necessary for the analysis, ensuring that the data frame is not cluttered with unnecessary information.

Finally, the cleaned data is stored in a structured CSV file, ready for exploratory data analysis and model training. The careful cleaning and structuring of data ensure that the insights derived from the model are accurate and reliable.

## Finalized Data Schema

The finalized data schema is detailed below, representing a comprehensive set of metrics calculated and cleaned from the raw game data. Each row corresponds to an individual player's performance in a game:

| Column Name | Description |
|-------------|-------------|
| `champ` | Champion name. |
| `position` | Player's position in the game. |
| `op_score` | Calculated score similar to op.gg's evaluation. |
| `win` | Binary indicator of game result (0 for loss, 1 for win). |
| `length` | Game length in minutes. |
| `kill` | Number of kills. |
| `death` | Number of deaths. |
| `assist` | Number of assists. |
| `kda` | Kill-Death-Assist ratio. |
| `dmg` | Total damage dealt to champions. |
| `magic_dmg` | Magic damage dealt to champions. |
| `ad_dmg` | Attack damage dealt to champions. |
| `all_dmg` | Total damage dealt. |
| `dmg_taken` | Total damage taken. |
| `ad_dmg_taken` | Attack damage taken. |
| `mitigated_dmg` | Damage self-mitigated. |
| `total_heal` | Total healing done. |
| `cs` | Total creep score. |
| `gold` | Gold earned. |
| `level` | Champion level at the end of the game. |
| `kp` | Kill participation percentage. |
| `dmg_perc` | Percentage of team's total damage dealt to champions. |
| `dmg_taken_perc` | Percentage of team's total damage taken. |
| `gold_perc` | Percentage of team's total gold earned. |
| `turret_kill` | Number of turret kills. |
| `inhib_kill` | Number of inhibitor kills. |
| `objective_dmg` | Damage dealt to objectives. |
| `turret_dmg` | Damage dealt to turrets. |
| `largest_multi_kill` | Largest multi-kill in the game. |
| `largest_kill_spree` | Largest killing spree. |
| `cc_score` | Crowd control score. |
| `dmg_per_gold` | Damage dealt per unit of gold earned. |
| `vision` | Vision score. |
| `pinks_bought` | Pink wards purchased. |
| `ward_kill` | Number of wards killed. |
| `ward_place` | Number of wards placed. |
| `cs_diff` | Differential in creep score compared to the opposing player in the same role. |
| `gold_diff` | Differential in gold earned compared to the opposing player in the same role. |
| `level_diff` | Differential in levels compared to the opposing player in the same role. |
| `dmg_taken_diff` | Differential in damage taken compared to the opposing player in the same role. |
| `dmg_diff` | Differential in damage dealt to champions compared to the opposing player in the same role. |

This table format provides a clear and concise snapshot of each player's performance metrics, enabling detailed analysis.

## Exploratory Data Analysis (EDA)

The Exploratory Data Analysis (EDA) section focuses on the mid lane data and involves multiple steps to understand the underlying patterns and relationships within the data. Below are the key components of my EDA process:

### Data Loading
Data is loaded from the processed CSV file specific to the mid lane. Initial examination of the data is performed using `data.head()` to visualize the first few rows.

### Statistical Summary
Descriptive statistics are calculated for each feature to provide insights into the central tendency, dispersion, and shape of the dataset's distribution.

### Potential Outliers
Outliers are identified using the Interquartile Range (IQR) method, enhanced by a factor to detect extreme values. This helps in understanding the data's variability and in making decisions about possible data cleaning.

### Correlation Analysis
Correlation matrices are plotted to identify relationships between different features. This includes:
- A standard correlation heatmap to view direct correlations.
- A clustered correlation heatmap to explore groups of correlated features, which might influence each other more significantly.

### Visualization of Feature Distributions
Histograms and box plots are generated for each feature to visualize the distribution and detect potential outliers, skewness, or other anomalies.

### Regression Analysis
Linear regressions are conducted to explore relationships:
- Between game length and other features to understand how metrics change over game duration.
- Between the OP score and other features to identify potential predictors of performance.

### Variance Inflation Factor (VIF)
VIF is calculated to check for multicollinearity among features, which could affect the performance of linear regression models.

### Visualizing Relationships
Various plots, including scatter plots of regressions and bar charts of VIF and correlations, are used to visually assess the strength and nature of relationships between features and the OP score.

This comprehensive EDA provides a robust foundation for subsequent model building and hypothesis testing, ensuring that the analysis is grounded in a thorough understanding of the data's characteristics.

## Initial Model Creation and Evaluation

The initial model creation explores various regression techniques to predict the OP score using the mid lane dataset. Each model's performance is assessed through metrics such as Mean Squared Error (MSE) and R-squared (R^2), providing insights into their predictive accuracy and explanatory power.

### Data Preparation

Data from the mid lane dataset undergoes preprocessing, including transformations and feature engineering, to optimize it for regression analysis.

### Model Evaluations

| Model                  | MSE      | R^2     | Details |
|------------------------|----------|---------|---------|
| **Initial Linear Regression** | 0.5498   | 0.8761  | Initial model serving as a baseline. |
| **Standardized Linear Regression** | 0.4938 | 0.8887  | Improved model with feature standardization. Coefficients for key features are listed below. |
| **Ridge Regression**   | 0.4937   | 0.8887  | Best alpha: 1. Features with coefficients close to zero include `ad_dmg_taken`. |
| **Lasso Regression**   | 0.4959   | 0.8882  | Best alpha: 0.001. Features removed: `magic_dmg`, `ad_dmg_taken`, and others. |
| **Elastic Net**        | 0.4950   | 0.8885  | Best alpha: 0.0007, l1_ratio: 1.0. Balances L1 and L2 penalties. |
| **Polynomial Regression** | 0.3650 | 0.9178  | Captures non-linear relationships effectively. |
| **Decision Tree**      | 0.7663   | 0.8273  | Shows potential overfitting with lower R^2. |
| **Random Forest**      | 0.3590   | 0.9191  | Best performing model with ensemble approach. |

### Model Comparison and Insights

- The range from simple linear regressions to complex ensemble methods highlights diverse approaches to tackling the prediction task.
- **Polynomial Regression** and **Random Forest** demonstrate the best performance, indicating their superior capability to handle the complex, non-linear interactions within the data.
- Decision Trees offer easy interpretability but may overfit, as shown by their lower R^2 value compared to ensemble methods.

This comprehensive modeling approach not only aids in accurate OP score prediction but also deepens the understanding of influential factors in mid-lane gameplay dynamics.

## Neural Network Implementation for OP Score Prediction in League of Legends

This section outlines the implementation of a neural network model designed to predict the op.gg game evaluation score for individual League of Legends games based on various performance metrics.

### Setup and Imports
- Essential libraries such as `TensorFlow`, `Keras`, and `Scikit-learn` are imported.
- An output directory for neural network models is set up: `../outputs/nn`.

### Data Loading and Preprocessing
- Data is loaded from `../data/processed/games.csv`.
- **Preprocessing Steps**:
  - Numerical features are normalized using `QuantileTransformer`.
  - Categorical features such as 'champ', 'position', and 'win' are encoded using `OneHotEncoder`.
  - A comprehensive `ColumnTransformer` is employed to apply these transformations.

### Model Architecture
- **Initial Model**: A simple neural network with layers designed to capture basic interactions in the data.
  - Architecture: Sequential model with dense layers.
  - Activation: ReLU for hidden layers and linear activation for the output layer.
  - Loss function: Mean squared error.

### Model Training and Evaluation
- **Data Splitting**: The dataset is divided into training and testing sets with a test size of 20%.
- **Training Process**:
  - The model is trained with validation to monitor performance and mitigate overfitting.
  - Training is performed over 50 epochs with a batch size of 32.

### Advanced Model Development
- A more complex model architecture is defined to handle different types of features effectively.
  - This includes separate subnetworks for game length and performance metrics, which are then combined.
  - Techniques such as batch normalization and dropout are used to improve training dynamics and prevent overfitting.

### Performance Metrics and Saving the Model
- **Evaluation**:
  - The model's performance is evaluated on the test set, and metrics like MSE and R^2 score are reported.
  - Results indicate that the model explains a significant portion of the variance in the data.
- **Model Saving**:
  - The trained model is saved to the specified output directory, ensuring that it can be reused or deployed.

### Model Reports and Visualization
- Training history is visualized to assess the learning process, with plots showing trends in loss and mean squared error over epochs.
- The final model shows improvements over the initial model, highlighting the benefits of the advanced architecture and training regimen.

### Summary of Key Results
- **Initial Model Results**:
  - **MSE**: 0.3237
  - **R^2 Score**: 0.9334
- **Advanced Model Results**:
  - **MSE**: 0.3156
  - **R^2 Score**: 0.9351

## Testing and Evaluating the Neural Network Models

The testing scripts are designed to load the trained neural network models and the preprocessing assets, make predictions on new data, and evaluate the model's performance. This section provides an overview of the key scripts and their functionalities.

### Model.py

This module contains functions for loading the necessary assets and making predictions. Here's a breakdown of its main components:

- **load_assets(preprocessor_path, model_path)**: Loads the preprocessor and the neural network model from specified paths.
- **make_predictions(data, preprocessor, model)**: Applies preprocessing to the input data and uses the model to generate predictions. This function supports models with multiple input layers, such as those separating game length from other performance metrics.

### Test_model.py

This script is used to evaluate the model using a separate test dataset. It includes functions for loading data, calculating performance metrics, and printing detailed results:

- **load_test_data(filepath)**: Loads test data from a CSV file.
- **evaluate_predictions(predictions, actuals)**: Calculates the Mean Squared Error (MSE) and R^2 Score to assess model accuracy and variance explanation.
- **print_detailed_results(predictions, actuals, data)**: Outputs a pretty-formatted table that lists predicted vs. actual OP scores along with other relevant game details such as win/loss, role, champion, game length, and KDA.

### Example Output

Upon executing the `test_model.py` script, you'll receive output similar to the following:

| Predicted OP Score | Actual OP Score | Win |  Role   |  Champ   | Length |  KDA   |
|--------------------|-----------------|-----|---------|----------|--------|--------|
|       3.216        |      3.604      |  1  |   TOP   |  Darius  | 15.617 | 0.857  |
|       7.292        |      7.445      |  1  |   ADC   |  Vayne   | 15.617 | 5.500  |
|       9.620        |      9.715      |  1  | SUPPORT |   Bard   | 15.617 | 17.000 |
|       9.678        |     10.000      |  1  |   MID   | Katarina | 15.617 | 20.000 |
|       7.325        |      7.806      |  1  | JUNGLE  |  Briar   | 15.617 | 3.500  |
|       1.798        |      2.680      |  0  |   ADC   |  Ezreal  | 15.617 | 0.500  |
|       5.064        |      5.598      |  0  |   TOP   |   Gnar   | 15.617 | 1.286  |
|       1.807        |      2.366      |  0  | SUPPORT |  Poppy   | 15.617 | 0.556  |
|       3.757        |      4.402      |  0  | JUNGLE  |  Viego   | 15.617 | 1.167  |
|       3.563        |      4.262      |  0  |   MID   |   Ahri   | 15.617 | 1.500  |

#### Global Metrics:
- **Mean Squared Error (MSE)**: 0.2799521775139535
- **R^2 Score**: 0.9602372408407295

This table and metrics provide a clear, tabulated representation of the predictions versus actual outcomes, along with other critical metrics to assess the model's accuracy and performance. The detailed results allow for easy verification of the model's effectiveness in different roles and game conditions.

## Dependencies

This project relies on various Python libraries across different stages of the data pipeline, including data scraping, cleaning, exploratory data analysis (EDA), model training, and testing. Below, you will find the required libraries for each section along with the pip install commands.

### Scraping
Libraries required for scraping data:
- pandas
- selenium
- BeautifulSoup4

```bash
pip install pandas selenium beautifulsoup4
```

**Note**: Selenium requires a WebDriver to interface with the chosen browser. Chrome, for example, requires `chromedriver`, which should be installed and path set accordingly.

### Cleaning
Libraries used for data cleaning:

- pandas

```bash
pip install pandas
```

### Exploratory Data Analysis (EDA)
Libraries used for EDA:

- pandas
- numpy
- matplotlib
- seaborn
- sklearn
- statsmodels

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

### Models
Libraries required for building and evaluating models:

- pandas
- numpy
- matplotlib
- seaborn
- sklearn
- scipy
- statsmodels
- tensorflow
- keras
- joblib

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels tensorflow keras joblib
```

### Model Testing
Libraries used for testing the models:

- pandas
- tabulate

```bash
pip install pandas tabulate
```

## Credits
All code and design was made by Eric Uehling.