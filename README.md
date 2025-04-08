# College Machine Learning Assignments in R

This repository contains two R-based machine learning assignments completed during college. These assignments demonstrate a range of data science skills including data preparation, model building, evaluation, feature selection, and dimensionality reduction using various R packages and techniques.

## Repository Structure

- **ML_CA1.R**  
  *Assignment 1 – Classification and Ensemble Learning*  
  - **Overview:**  
    This assignment focuses on classification tasks using different approaches:
    - **LDA and QDA:**  
      - Prepares the CA dataset.
      - Fits Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) models.
      - Compares model performance via confusion matrices and computes specificity for each model.
    - **Random Forest and GBM on Khan’s Data:**  
      - Uses the Khan dataset (from the `ISLR` package) for a classification problem.
      - Trains a Random Forest model with cross-validation, computes prediction accuracy, and extracts feature importance.
      - Applies Gradient Boosting Machines (GBM) to further predict outcomes and evaluate model performance.
      
- **ML_CA2.R**  
  *Assignment 2 – k-NN, Feature Selection, Neural Networks & PCA*  
  - **Overview:**  
    This assignment is structured into several parts:
    - **(Q2) k-NN Modeling and Feature Selection:**  
      - Reads training and validation datasets.
      - Employs k-Nearest Neighbors (k-NN) with cross-validation to determine the optimal number of neighbors.
      - Uses Recursive Feature Elimination (RFE) with Random Forest (RF) to select important features.
      - Applies a neural network model (using the `nnet` package) to predict class labels on the validation set.
    - **(Q3) Principal Component Analysis (PCA) and Regression:**  
      - Performs PCA on the built-in `mtcars` dataset.
      - Compares eigenvectors obtained from scaled and non-scaled data.
      - Computes the proportion of variance explained (PVE) and identifies the top variables.
      - Fits a linear regression model using the top five variables to predict `mpg` and reviews model performance.

## Prerequisites

- **R (version 3.6 or later)** – It is recommended to use [RStudio](https://www.rstudio.com/) for the best development experience.
- **Required R Packages:**  
  The scripts depend on a number of packages. Here are some of the key ones:
  - **ML_CA1.R:** `MASS`, `caret`, `randomForest`, `gbm`, `ISLR`
  - **ML_CA2.R:** `caret`, `randomForest`, `nnet`  
   
  You can install these packages (if not already installed) using:
  
  ```R
  install.packages(c("MASS", "caret", "randomForest", "gbm", "ISLR", "nnet"))
  ```

## Running the Assignments

1. **Clone or Download the Repository:**

   ```bash
   git clone https://github.com/YourUsername/CollegeMLAssignmentsR.git
   cd CollegeMLAssignmentsR
   ```

2. **Open the R Scripts in RStudio:**

   - Launch RStudio and open the scripts `ML_CA1.R` and `ML_CA2.R`.
   - Update the file paths within the scripts as necessary, especially if your data files are stored in different locations.

3. **Execute the Scripts:**

   - You can run each script by clicking the "Run" button inside RStudio, or from the command line:
   
     ```bash
     Rscript ML_CA1.R
     Rscript ML_CA2.R
     ```

## Assignment Details

### ML_CA1.R – Classification and Ensemble Models
- **Data Preparation:**  
  Loads CA train and test datasets, selects predictors, and defines the response variable.
- **Modeling Approaches:**  
  Implements LDA and QDA models to assess classification performance via confusion matrices and specificity comparisons.
- **Ensemble Methods on Khan Data:**  
  Utilizes Random Forests (with visualizations and variable importance) and GBM for classification tasks on the Khan dataset, further evaluating performance with accuracy metrics.

### ML_CA2.R – k-NN, Feature Selection & PCA
- **k-NN and Feature Selection:**  
  - Uses cross-validation with k-NN to select the optimal number of neighbors.
  - Applies Recursive Feature Elimination with a Random Forest to identify influential features.
  - Implements a neural network model to evaluate predictive performance on a validation set.
- **Principal Component Analysis (PCA) and Regression:**  
  - Performs PCA on the `mtcars` dataset to compare the effect of scaling.
  - Computes the proportion of variance explained and determines key variables.
  - Fits a linear regression model using the top five PCA variables to predict `mpg`.

## Acknowledgments

- Thanks to the instructors and peers who provided guidance during these assignments.
- Special thanks to the developers of R and the associated packages (MASS, caret, randomForest, gbm, etc.) that made these analyses possible.
