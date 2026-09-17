# Detailed Comparison of kNN and Logistic Regression

## Evaluation framework
The comparison framework used three controls to keep the benchmark fair and interpretable:

1. **Identical input data.** Both models used the same 7,043-customer Telco churn data set after identical cleaning steps.
2. **Identical preprocessing.** Numeric fields were imputed and standardized, while categorical fields were imputed and one-hot encoded in the same shared pipeline.
3. **Identical resampling design.** Both models were trained on the same 80/20 stratified split and assessed with 5-fold stratified cross-validation on the training data.

This framework follows standard supervised-learning practice because it separates model selection from final holdout evaluation and reduces the risk of judging models on a single accidental split (James et al., 2021; Kuhn & Johnson, 2013).

## Why these metrics were used
- **Accuracy** summarizes overall correctness but can be misleading when classes are imbalanced.
- **Precision** measures how often predicted churn cases are truly churners.
- **Recall** measures how many true churners are captured.
- **F1-score** balances precision and recall and was therefore used as the primary tuning criterion.
- **ROC-AUC** measures ranking quality across thresholds and is useful for comparing overall discrimination (Hastie et al., 2009).

## Model results
### Cross-validation means
| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| kNN | 0.7943 | 0.6167 | 0.5933 | 0.6046 | 0.8340 |
| Logistic regression | 0.8021 | 0.6530 | 0.5431 | 0.5923 | 0.8461 |

### Holdout test results
| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| kNN | 0.7842 | 0.5956 | 0.5829 | 0.5892 | 0.8268 |
| Logistic regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8419 |

## What the comparison shows
### 1. Logistic regression had the best overall test performance
The logistic regression model won on four of the five holdout metrics: accuracy, precision, F1-score, and ROC-AUC. That makes it the best overall model under this framework because it combined better discrimination with a better balance between false positives and false negatives.

### 2. kNN had slightly stronger recall
kNN recovered 58.3% of churners, compared with 55.9% for logistic regression. If the business goal were to identify as many churners as possible and accept a few more false alarms, that recall edge could justify using kNN or adjusting the logistic-regression decision threshold.

### 3. Cross-validation and test results tell a nuanced story
During cross-validation, kNN posted the stronger mean F1-score, while logistic regression produced the stronger mean ROC-AUC and precision. On the unseen test set, logistic regression moved ahead on F1-score as well. That shift suggests logistic regression generalized a little more reliably to new customers in this run.

### 4. Model behavior differs conceptually
- **kNN** is instance-based and flexible. It can model nonlinear local patterns, but it is sensitive to scaling, sparse features, and neighborhood definition.
- **Logistic regression** is a parametric linear classifier. It is usually easier to interpret, faster to score, and often more stable on structured tabular data when the class boundary is not highly irregular (James et al., 2021).

## Which model performs better?
Using this project’s evaluation framework, **logistic regression performs better overall**. The deciding criteria were:
- higher holdout **accuracy**;
- higher holdout **precision**;
- higher holdout **F1-score**;
- higher holdout **ROC-AUC**.

The only metric favoring kNN was recall, and the margin was small. Therefore, the best evidence-supported conclusion is that logistic regression is the preferred model for the present Telco churn classification task, while kNN remains a useful secondary benchmark.

## References
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning*.
- Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling*.
