# Detailed Comparison of kNN and Logistic Regression

## Comparison objective

This document provides a focused comparison of the two churn classifiers built from the same telecom customer data: a tuned k-Nearest Neighbors model and a logistic regression model. The comparison uses the exact same train/test split, the same missing-data handling, the same feature scaling for numeric inputs, and the same one-hot encoding for categorical inputs. Holding the data pipeline constant makes the model contrast methodologically fair.

## 1. How the two models differ conceptually

### kNN

kNN is an instance-based, non-parametric classifier. It does not estimate a global equation during training. Instead, it stores the training data and classifies each new customer from the majority vote of the nearest training neighbors. Its main strength is flexibility: it can model nonlinear local patterns without assuming a fixed functional form. Its main weakness is that it depends heavily on distance quality, feature scaling, and neighborhood structure (Cunningham & Delany, 2021; Zhang, 2016).

### Logistic regression

Logistic regression is a parametric classifier that models the log-odds of churn as a function of the predictors. It assumes that the transformed predictors can be separated by a linear decision rule in log-odds space. Its main strengths are interpretability, stability, and efficient scoring. Its main weakness is that it may underfit when the true decision boundary is strongly nonlinear (Bewick et al., 2005; Harris, 2021).

## 2. Shared evaluation framework

The comparison used the following framework:

- **Data split:** 80% training, 20% testing, stratified by churn status
- **Preprocessing:** median imputation for numeric missingness, standardization for numeric predictors, most-frequent imputation for categorical missingness, and one-hot encoding for categorical variables
- **Model selection:** 5-fold stratified cross-validation for kNN, with ROC AUC as the tuning metric
- **Assessment metrics:** accuracy, precision, recall, F1, and ROC AUC

This framework is appropriate because no single metric fully describes classifier performance on an imbalanced outcome. Accuracy summarizes overall correctness, precision measures the reliability of positive churn predictions, recall measures how many true churners were found, F1 balances precision and recall, and ROC AUC evaluates ranking quality across thresholds (Fawcett, 2006; Géron, 2022).

## 3. Empirical results

| Metric | kNN | Logistic regression | Better model |
| --- | ---: | ---: | --- |
| Accuracy | 0.785 | **0.805** | Logistic regression |
| Precision | 0.599 | **0.655** | Logistic regression |
| Recall | **0.575** | 0.559 | kNN |
| F1 | 0.587 | **0.603** | Logistic regression |
| ROC AUC | 0.827 | **0.841** | Logistic regression |

The kNN model selected **k = 25**, which indicates that smoother neighborhoods worked better than small, highly local ones. The cross-validated ROC AUC for the selected kNN configuration was **0.832**. Logistic regression, evaluated on the same holdout set, produced stronger discrimination overall.

## 4. Interpretation of the differences

### Why logistic regression won overall

Logistic regression performed better on four of the five reported metrics. This suggests that the churn signal in the transformed telecom data is sufficiently structured for a regularized linear classifier to capture the dominant patterns. In practical terms, logistic regression generated cleaner separation between churners and non-churners and reduced false alarms compared with kNN.

### Why kNN still remained competitive

kNN slightly outperformed logistic regression on recall. That means it found a few more churners, even though it did so less precisely. This result makes sense because kNN can adapt to local pockets of similar churn behavior that a single linear boundary may smooth over. However, the gain was small and came with lower precision, lower overall accuracy, and lower ROC AUC.

### Operational implications

- **Interpretability:** logistic regression is easier to explain to stakeholders because predictor effects can be summarized through coefficients and odds ratios, while kNN decisions are case-based and less transparent.
- **Scalability:** logistic regression is faster at prediction time because it uses a fixed coefficient vector, whereas kNN must compare new customers with the stored training cases.
- **Robustness in this data set:** after one-hot encoding the categorical predictors, the feature space becomes wider and sparser, which generally makes distance-based methods harder to stabilize than a linear probabilistic classifier.

## 5. Final recommendation

If this telecom company needs one model from the two tested approaches, logistic regression is the better choice for the current assignment. It delivered the best balance of discrimination, precision, and overall accuracy while remaining simpler to interpret and operationalize. kNN is still useful as a comparison benchmark because it confirms that local customer similarity contains real churn information, but it is not the strongest overall performer on this data.

## References

- Bewick, V., Cheek, L., & Ball, J. (2005). Statistics review 14: Logistic regression. *Critical Care, 9*(1), 112-118. https://pmc.ncbi.nlm.nih.gov/articles/PMC1065119/
- Cunningham, P., & Delany, S. J. (2021). k-Nearest neighbour classifiers: A tutorial. *ACM Computing Surveys, 54*(6), Article 128. https://doi.org/10.1145/3459665
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861-874. https://doi.org/10.1016/j.patrec.2005.10.010
- Géron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
- Harris, J. K. (2021). Primer on binary logistic regression. *Family Medicine and Community Health, 9*, e001290. https://fmch.bmj.com/content/9/4/e001290
- Zhang, Z. (2016). Introduction to machine learning: k-nearest neighbors. *Annals of Translational Medicine, 4*(11), 218. https://pmc.ncbi.nlm.nih.gov/articles/PMC4916348/
