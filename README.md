## College Name - Team Name
**College Name :** Anna University Regional Campus Coimbatore   
**Team Name :** NJRHACKS

## Problem Statement
**Theme:** Quantum Computing  
 **Problem Statement:** *Quantum Detective:Cracking Financial Anomalies*  
  Traditional Machine Learning Models often face challenges like fraudlent activities,unusual anomolies due to their size and complexity resulting in less accurate result and slower processing .To overcome  these challenges we use quantum properties to analyse  large dataset more efficiently.  
  Goal : To design the quantum model that demonstrates improvement in Speed ,Scalability and Accuracy when compared to Classical methods .
  
### Instructions on running your project
>Instructions on running our project
#### Classical Computing
#### Overview
In this notebook, exploring various Machine Learning models to detect fraudulent use of credit cards. and comparing each model's performance and results. The best performance is achieved using the XGBOOST technique.
#### Techniques Used in the Project
The project compares the results of different techniques:

#### Machine Learning Techniques
Decision Trees
XGBOOST

### Note about Random Forest and Decision Tree Models:
Decision Tree: Built on an entire dataset using all features/variables. You can easily overfit the data, so it is recommended to use cross-validation. Advantages: easy to interpret, clear understanding of the variable and value used for splitting data and predicting outcomes.

XGBOOST: XGBoost is an efficient and scalable implementation of gradient boosting framework, designed for speed and performance. It builds an ensemble of decision trees one at a time, where each new tree corrects the errors made by the previously built trees.

###  Result
TEST RESULTS
test set:
Accuracy: 0.99
    Sensitivity (Recall): 0.72
    Specificity: 0.99
    F1-Score: 0.71
    ROC-AUC: 0.97

Training set:
    Accuracy: 1.0
    Sensitivity (Recall): 1.0
    Specificity: 1.0
    F1-Score: 1.0
    ROC-AUC: 1.0


##### Conclusion

This project demonstrates the effectiveness of machine learning  techniques in detecting fraudulent credit card transactions. The use of the desicion tree and XGBOOST technique significantly improves the detection of fraudulent transactions, making the system robust and reliable for real-time updates.

  
## Quantum Computing
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
  ## Comparision between classical and quantum
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
## Architetcture Model
quantum detective:cracking financial anomalies
Here's a simplified geometric representation of your project structure, focusing on the main components without excessive detail:

### Simplified Geometric Representation


┌────────────────────┐
│     Start          │
└──────────┬─────────┘
           ▼
┌────────────────────┐
│   Classical        │
│   Computing        │
└──────────┬─────────┘
           ▼
┌────────────────────┐
│   Data             │
│   Exploration      │
└──────────┬─────────┘
           ▼
┌────────────────────┐
│   Data             │
│   Visualization     │
└──────────┬─────────┘
           ▼
┌────────────────────┐
│   Data             │
│   Preparation      │
└──────────┬─────────┘
           ▼
┌────────────────────┐
│   Model            │
│   Training         │
└──────────┬─────────┘
           ▼
┌────────────────────┐
│   Quantum          │
│   Computing        │
└──────────┬─────────┘
           ▼
┌────────────────────┐
│   Data             │
│   Preparation      │
└──────────┬─────────┘
           ▼
┌────────────────────┐
│   Model            │
│   Training         │
└──────────┬─────────┘
           ▼
┌────────────────────┐
│   Comparison       │
│   of Models        │
└──────────┬─────────┘
           ▼
┌────────────────────┐
│   Conclusion       │
└────────────────────┘


### Key Features of this Simplified Representation:
- **High-Level Structure**: Focuses on essential sections of your project without diving into subcomponents.
- **Clear Flow**: Arrows indicate the progression from one main category to another, making it easy to understand the sequence of tasks.
- **Balanced Sections**: Both classical and quantum computing components are represented, emphasizing their importance in the overall project.

### Conclusion
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
5.  "youtube"
