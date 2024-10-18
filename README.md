## College Name - Team Name
**College Name :** Anna University Regional Campus Coimbatore   
**Team Name :** NJRHACKS

## Problem Statement
**Theme:** Quantum Computing  
 **Problem Statement:** *Quantum Detective:Cracking Financial Anomalies*  
  Traditional Machine Learning Models often face challenges like fraudlent activities,unusual anomolies due to their size and complexity resulting in less accurate result and slower processing .To overcome  these challenges we use quantum properties to analyse  large dataset more efficiently.  
  Goal : To design the quantum model that demonstrates improvement in Speed ,Scalability and Accuracy when compared to Classical methods .
  
## Instructions on running your project
>Instructions on running our project
###### Classical Computing
- **Libraries and Data Loading:**
1. Imports necessary libraries such as NumPy, Pandas, Matplotlib, Seaborn, and various machine learning algorithms from Scikit-learn and XGBoost.
2. Loads the dataset (creditcard.csv) into a Pandas DataFrame.
- **Initial Data Exploration:**
1. Displays the last few rows, summary statistics, and info about the dataset.
2. Checks for missing values and computes the percentage of null entries in each column.
3. Analyzes the distribution of classes (fraudulent vs. non-fraudulent transactions).
- **Data Visualization:**
1. Creates bar plots to visualize the number and percentage of fraudulent vs. non-fraudulent transactions.
2. Displays a boxplot to explore the distribution of transaction amounts.
3. Uses kernel density estimation (KDE) plots to compare transaction times for both classes.
4. Generates a heatmap to visualize the correlation matrix of features.
- **Data Preparation:**
1. Splits the data into features (X) and target variable (y).
2. Performs a train-test split, with 80% of the data for training and 20% for testing.
3. Scales the Amount feature using StandardScaler.
- **Model Training:**
1. Trains a Decision Tree classifier on the training set.
2. Makes predictions on the test set and evaluates the model using metrics such as accuracy, F1 score, and a confusion matrix.
3. Draws ROC curves for both training and test sets to visualize model performance.
- **XGBoost Model:**
1. Trains an XGBoost classifier and evaluates its performance similarly.
2. Prints classification reports and confusion matrices for comparison.  
###### Quantum Computing
- **Library Imports:** Imports essential libraries for data handling, visualization, machine learning, and quantum computing.
- **Data Loading and Exploration:** Loads a credit card dataset and visualizes the distribution of features and their correlations.
- **Data Preparation:** Splits the dataset into normal and fraudulent transactions.
Balances the dataset by sampling equal numbers of normal and fraudulent cases.
  the features to the range [0, 1] and applies zero padding to ensure the number of features is a power of 2.
- **Train-Test Split:** Divides the dataset into training and testing sets for model evaluation.
- **Quantum Circuit Definition:** Sets up a quantum device and defines a variational circuit using a feature map and parameterized gates to encode classical data into quantum states.
- **Cost Function:** Defines a cost function to compute the mean squared error between the predicted and actual labels.
- **Model Training:** Uses an optimizer to train the variational quantum classifier over multiple epochs, updating the parameters based on the cost function.
- **Model Evaluation:** Makes predictions on the test set and evaluates the classifier's performance using accuracy and a classification report.
  ###### Comparision between classical and quantum
- **Library Imports:**
1. Imports necessary libraries such as NumPy, Pandas, and Scikit-learn for machine learning and evaluation metrics.
2. Imports Matplotlib for visualization and Joblib for model saving (though it's not used in the code).
- **Data Generation:**
1. A synthetic dataset is created in the load_data() function, simulating 10,000 samples with 10 features and a class imbalance (1% fraud).
2. Mock Quantum Model:
3. Defines a MockQuantumModel class that simulates the behavior of a quantum model. It has methods for fitting, predicting, and predicting probabilities, but the actual functionality is mocked.
**Model Evaluation:**
   The evaluate_model() function calculates and prints various performance metrics (accuracy, precision, recall, F1 score, and AUC) for a given model. It uses the model's predict() and predict_proba() methods to get predictions and probabilities.
- **Model Comparison Visualization:**
    The compare_models() function creates a bar chart comparing the performance metrics of the classical and quantum models using Matplotlib.
- **Main Execution Flow:**
The main() function orchestrates the overall process:
1. Loads synthetic data.
2. Splits the data into training and test sets.
3. Scales the feature values using StandardScaler.
4. Trains the classical model (Decision Tree) and the mock quantum model.
5. Evaluates both models and compares their performance metrics.
- **Script Execution:**
The script checks if it is run as the main program and executes the main() function.
###### Conclusion
- **Effectiveness of Classical Models:** Classical models, exemplified by the Decision Tree Classifier, demonstrate strong performance in fraud detection, achieving impressive metrics in accuracy, precision, recall, and F1 score. These results validate the effectiveness of traditional machine learning techniques for analyzing large, imbalanced datasets typical of credit card transactions.
- **Quantum Computing Potential:** While the quantum model presented a theoretical framework, it suggests the promise of quantum computing in capturing complex patterns within data that classical models may overlook. However, practical applications remain constrained by current technological limitations.
- **Scalability and Practicality:** Classical machine learning models excel in scalability and efficiency, making them highly suitable for real-time fraud detection systems. Conversely, quantum computing faces challenges such as noise, qubit coherence, and the need for further refinement of algorithms to effectively handle large-scale data.
- **Future Outlook:** The comparison highlights the necessity for continued research in quantum machine learning to develop robust algorithms capable of outperforming classical methods. Exploring hybrid models that combine classical and quantum techniques may yield innovative solutions for complex fraud detection problems.
- **Recommendations:** For immediate implementation, classical models are the most viable choice for credit card fraud detection. However, investing in quantum research could offer significant long-term benefits as the technology matures. Collaboration between researchers in both fields could accelerate the development of effective quantum solutions.
  
## References
1. "Qiskit: An Open-source Quantum Computing Framework." Retrieved from [Qiskit](https://learning.quantum.ibm.com/)
2. "PennyLane: A Python Library for Quantum Machine Learning." Retrieved from [PennyLane](https://pennylane.ai/)
3. "Credit Card Fraud Detection." Kaggle Dataset retrieved from [kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
4. "Quantum Computing for Financial Applications: A Survey," IEEE Quantum Electronics[DOI:10.1109/QE.2022.09915517](https://www.computer.org/csdl/journal/qe/2022/01/09915517/1HmgdJyXCqQ)
