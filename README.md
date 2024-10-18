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
**Import Libraries:** The code begins by importing necessary libraries for data manipulation (NumPy, Pandas), machine learning preprocessing and evaluation (Scikit-learn), and quantum computing (Qiskit).
**Load and Sample the Dataset:** The dataset is loaded from a CSV file into a DataFrame.
A random sample of 1000 data points is taken from the dataset to ensure the size is manageable for quantum processing.  
**Separate Features and Labels:** Features (input variables) and labels (target variable) are separated. The target variable indicates whether a transaction is fraudulent or not. 
**Feature Scaling:** The features are standardized using StandardScaler to normalize their values, which is important for effective model training.  
**Train-Test Split:** The dataset is split into training and testing sets (80% for training, 20% for testing) to evaluate the model's performance.
**Define the Quantum Feature Map:** A quantum feature map (ZZFeatureMap) is created to transform classical data into a quantum representation. This maps the input features into a quantum state.    
**Set Up Quantum Instance:** A quantum instance is defined using a simulator backend (in this case, the qasm_simulator), specifying the number of shots (repetitions) for measurement accuracy.    
**Create the Quantum Kernel:** A quantum kernel is initialized with the defined feature map and quantum instance. This kernel will facilitate the training of the QSVM.  
**Initialize and Train the QSVM Model:** An instance of the QSVM is created using the quantum kernel, and the model is trained on the training data.  
**Model Testing:** The trained QSVM model is used to make predictions on the test set.  
**Model Evaluation:** The performance of the QSVM is evaluated using a classification report that provides metrics such as precision, recall, and F1-score.

## References
1. "Qiskit: An Open-source Quantum Computing Framework." Retrieved from [Qiskit](https://learning.quantum.ibm.com/)
2. "PennyLane: A Python Library for Quantum Machine Learning." Retrieved from [PennyLane](https://pennylane.ai/)
3. "Credit Card Fraud Detection." Kaggle Dataset retrieved from [kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
4. "Quantum Computing for Financial Applications: A Survey," IEEE Quantum Electronics[DOI:10.1109/QE.2022.09915517](https://www.computer.org/csdl/journal/qe/2022/01/09915517/1HmgdJyXCqQ)
