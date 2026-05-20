# Customer Retention Prediction using Deep Learning

## Project Overview

This project uses a deep learning approach to predict customer retention using a customer retention dataset. The main goal is to build, train, and compare different neural network configurations for binary classification, where the target variable is **Retention**.

The notebook focuses on preprocessing customer data, handling categorical features, scaling numerical values, training neural networks, and comparing model performance using different activation functions and gradient descent strategies.

## Dataset

The dataset used in this project is:

**Customer Retention Dataset.csv**

In the notebook, the dataset is loaded from Kaggle using the following path:

```python
/kaggle/input/customer-retention-dataset/Customer Retention Dataset.csv
```

The target column is:

```python
Retention
```

All other columns are used as input features.

## Objectives

The main objectives of this project are:

- Load and preprocess the customer retention dataset
- Separate features and target variable
- Handle categorical variables using one-hot encoding
- Scale the feature values using StandardScaler
- Build neural network models for binary classification
- Compare Sigmoid and ReLU-based models
- Compare different gradient descent approaches
- Compare multiple activation functions including Sigmoid, Tanh, ReLU, and Leaky ReLU
- Evaluate model performance using accuracy and loss

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras

## Project Workflow

### 1. Import Libraries

The project starts by importing the required Python libraries for data processing, visualization, machine learning, and deep learning.

### 2. Load Dataset

The dataset is loaded using Pandas:

```python
data = pd.read_csv("/kaggle/input/customer-retention-dataset/Customer Retention Dataset.csv")
```

### 3. Feature and Target Split

The target column is separated from the input features:

```python
X = data.drop("Retention", axis=1)
y = data["Retention"]
```

### 4. Categorical Encoding

Categorical features are converted into numerical form using one-hot encoding:

```python
X = pd.get_dummies(X, drop_first=True)
```

### 5. Train-Test Split

The dataset is divided into training and testing sets. Stratified splitting is used in later experiments to preserve the class distribution.

### 6. Feature Scaling

Feature values are standardized using `StandardScaler` to improve neural network training.

### 7. Model Building

Several neural network models are created using Keras Sequential API. The models use dense layers and are trained for binary classification with a sigmoid output layer.

## Experiments Performed

### Experiment 1: Sigmoid Model vs ReLU Model

Two neural network models were trained and compared:

| Model | Test Accuracy | Test Loss |
|---|---:|---:|
| Sigmoid Model | 65.78% | 0.6426 |
| ReLU Model | 59.92% | 0.6976 |

The Sigmoid model performed better than the ReLU model in this experiment.

### Experiment 2: Gradient Descent Comparison

Three training strategies were compared:

| Method | Batch Size | Training Time (sec) | Final Validation Accuracy | Test Accuracy |
|---|---:|---:|---:|---:|
| SGD standard | 32 | 181.76 | 65.44% | 65.36% |
| Batch Gradient Descent | 40000 | 9.57 | 63.90% | 63.01% |
| Mini-Batch Gradient Descent | 32 | 178.04 | 65.44% | 65.36% |

SGD and Mini-Batch Gradient Descent achieved the best test accuracy, while Batch Gradient Descent was faster but less accurate.

### Experiment 3: Activation Function Comparison

Four activation functions were compared using a neural network with three hidden layers:

| Activation Function | Training Time (sec) | Final Training Accuracy | Final Validation Accuracy | Final Validation Loss |
|---|---:|---:|---:|---:|
| Sigmoid | 181.15 | 65.45% | 65.06% | 0.6493 |
| Tanh | 182.32 | 68.27% | 60.85% | 0.6977 |
| ReLU | 185.03 | 78.48% | 57.40% | 0.9590 |
| Leaky ReLU | 186.71 | 77.63% | 55.62% | 0.9542 |

Based on final validation accuracy, the best activation function was:

```text
Sigmoid
```

## Results Summary

The best overall result in the notebook was achieved by the **Sigmoid activation function**, with a final validation accuracy of **65.06%**.

The comparison shows that although ReLU and Leaky ReLU achieved higher training accuracy, their validation accuracy was lower. This suggests possible overfitting or poor generalization on the validation data.

## Key Findings

- Sigmoid performed better than ReLU for this customer retention prediction task.
- Standard SGD and Mini-Batch Gradient Descent achieved similar test accuracy.
- Batch Gradient Descent trained faster but produced lower test accuracy.
- ReLU and Leaky ReLU showed higher training accuracy but weaker validation performance.
- The model's accuracy stayed around 65%, which means further improvement may require better feature engineering, hyperparameter tuning, or trying other machine learning models.

## How to Run the Project

### Option 1: Run on Kaggle

1. Upload the notebook to Kaggle.
2. Add the Customer Retention Dataset to the notebook.
3. Make sure the dataset path matches:

```python
/kaggle/input/customer-retention-dataset/Customer Retention Dataset.csv
```

4. Run all cells from top to bottom.

### Option 2: Run Locally

1. Clone this repository:

```bash
git clone https://github.com/your-username/customer-retention-prediction.git
```

2. Move into the project folder:

```bash
cd customer-retention-prediction
```

3. Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

4. Place the dataset file in the project directory.

5. Update the dataset path in the notebook:

```python
data = pd.read_csv("Customer Retention Dataset.csv")
```

6. Run the notebook using Jupyter Notebook or JupyterLab.

## Project Structure

```text
customer-retention-prediction/
│
├── customer_retention.ipynb
├── Customer Retention Dataset.csv
└── README.md
```

## Future Improvements

- Perform exploratory data analysis before training
- Check class balance in the target variable
- Add confusion matrix, precision, recall, and F1-score
- Use ROC-AUC for better binary classification evaluation
- Apply dropout or regularization to reduce overfitting
- Try more models such as Logistic Regression, Random Forest, XGBoost, and CatBoost
- Perform hyperparameter tuning
- Use early stopping during model training

## Conclusion

This project demonstrates how deep learning can be used for customer retention prediction. Multiple neural network experiments were performed to compare activation functions and optimization approaches. The results show that the Sigmoid-based model provided the best validation performance in this notebook, making it the most suitable model among the tested configurations.
