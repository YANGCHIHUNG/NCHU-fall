# Titanic Survival Prediction

This project demonstrates the use of machine learning techniques to predict the survival of passengers on the Titanic. The project is divided into two parts, each represented by a Jupyter notebook.

## Notebooks

### HW4-1.ipynb

This notebook focuses on the initial data exploration, preprocessing, and model comparison using PyCaret.

#### Steps:
1. **Import Libraries**: Import necessary libraries such as `pandas`, `seaborn`, `matplotlib`, and `pycaret.classification`.
2. **Load Dataset**: Load the Titanic dataset and display its structure and summary statistics.
3. **Data Preprocessing**: Drop unnecessary columns and split the data into training and testing sets.
4. **Setup PyCaret**: Initialize the PyCaret setup with the training data.
5. **Compare Models**: Use PyCaret to compare different classification models and select the best ones.

### HW4-2.ipynb

This notebook extends the work done in HW4-1.ipynb by focusing on model creation, hyperparameter tuning using Optuna, and model evaluation.

#### Steps:
1. **Import Libraries**: Import necessary libraries such as `pandas`, `seaborn`, `matplotlib`, `pycaret.classification`, and `optuna`.
2. **Load Dataset**: Load the Titanic dataset and display its structure and summary statistics.
3. **Data Preprocessing**: Drop unnecessary columns and split the data into training and testing sets.
4. **Setup PyCaret**: Initialize the PyCaret setup with the training data.
5. **Compare Models**: Use PyCaret to compare different classification models and select the best ones.
6. **Create Model**: Create a Random Forest model.
7. **Hyperparameter Tuning with Optuna**: Define an objective function for Optuna to optimize the hyperparameters of the Random Forest model.
8. **Optimize Model**: Use Optuna to find the best hyperparameters and create a tuned model.
9. **Evaluate Model**: Evaluate the tuned model using PyCaret's evaluation tools.
10. **Save Model**: Save the tuned model to a file.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/YANGCHIHUNG/IoT-data-analysis-and-application.git
    cd HW4
    ```

2. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run Jupyter Notebook**:
    ```sh
    jupyter notebook
    ```

2. **Open the notebooks**:
    - Open `HW4-1.ipynb` to explore the data and compare models.
    - Open `HW4-2.ipynb` to create, tune, and evaluate the model.

## Dataset

The dataset used in this project is the Titanic dataset, which can be found in the `dataset/titanic/` directory.

## Dependencies

- pandas
- seaborn
- matplotlib
- scikit-learn
- pycaret
- optuna

## License

This project is licensed under the MIT License.