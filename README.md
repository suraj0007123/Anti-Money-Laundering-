Data Exploration and Preprocessing:

Data is loaded using pandas, and basic information about the dataset is displayed.
Measures of central tendency (mean, median, mode) and dispersion (variance) are calculated.
Skewness and kurtosis are also computed.
Some columns with missing values are filled using median, mode, or mean values.
Duplicate values are checked, and none are found.
New features 'high' and 'rapid' are created based on transaction amount and beneficiary account frequency, respectively.
Feature Engineering:

A new feature 'merchant' is created based on conditions involving transaction types and receiver names.
One-hot encoding is applied to the 'type' column.
Customer names ('nameOrig' and 'nameDest') are dropped from the dataset.
Numerical columns are normalized.
Model Building:

The dataset is split into features (X) and the target variable (Y).
The data is split into training and testing sets.
The class imbalance is addressed using the Synthetic Minority Over-sampling Technique (SMOTE) to oversample the minority class.
A Decision Tree Classifier is trained on the balanced dataset.
Model evaluation metrics such as accuracy, confusion matrix, and F1 score are computed for both the training and testing sets.
Model Serialization:

The trained Decision Tree Classifier is saved using the pickle library.
Insights and Recommendations:

The model achieves a high accuracy on both the training and testing sets, suggesting that it has learned well from the data.
The use of SMOTE indicates an awareness of class imbalance and an attempt to address it.
The decision tree model may be interpretable, but it's worth considering more complex models for comparison.
Evaluation metrics such as precision, recall, and ROC-AUC could provide a more comprehensive view of the model's performance, especially given the imbalanced nature of the dataset.
Further analysis, such as feature importance, could provide insights into which features contribute the most to fraud detection
