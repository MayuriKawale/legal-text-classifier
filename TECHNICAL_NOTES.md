# Executive Summary

## What This Repository Does

This project builds an automated system to identify potential human rights 
violations in legal case documents. Given the facts of a case from the 
European Court of Human Rights (ECtHR), the model predicts which articles 
of the European Convention on Human Rights (ECHR) were violated, a task 
that would otherwise require significant manual review by legal experts.

---

## How Was the Data Labeled?

The data was not self-labeled. Labels were assigned by the ECtHR itself. 
Each case in the dataset corresponds to a real court judgment where the court 
officially determined which ECHR articles were violated. These ground-truth 
labels were compiled into the LexGLUE benchmark by Chalkidis et al. (2022), 
a peer-reviewed legal NLP benchmark widely used in the research community. 
This makes the labels highly reliable as they reflect actual judicial decisions 
rather than human annotation.

---

## How Was the Model Tested?

The model was evaluated on 1,000 held-out test cases that were never seen 
during training. This ensures that reported performance reflects how well the 
model generalizes to new, unseen legal text rather than simply memorizing the 
training data.

---

## Evaluation Metrics

### Why Not Accuracy?

**Accuracy**: the percentage of correct predictions overall, is misleading 
for this dataset due to severe class imbalance. For example, Article 9 appears 
in only 0.46% of cases. A model that never predicts Article 9 would still 
achieve 99.54% accuracy on that label while being completely useless. Accuracy 
rewards models for simply predicting the majority class.

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Where:
- **TP** = True Positives — correctly predicted violations
- **TN** = True Negatives — correctly predicted non-violations
- **FP** = False Positives — predicted violation when there was none
- **FN** = False Negatives — missed an actual violation

---

### Precision and Recall

**Precision**: measures how often the model is correct when it predicts a 
violation, out of all cases the model flagged as a violation, how many 
actually were? High precision means few false alarms.

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall**: measures how often the model catches actual violations, out of 
all real violations, how many did the model find? High recall means few 
missed violations.

$$\text{Recall} = \frac{TP}{TP + FN}$$

Precision and recall are often in tension; improving one tends to hurt the 
other. A model that predicts every case as a violation will have perfect recall 
but very low precision. A model that only predicts violations when highly 
confident will have high precision but low recall.

---

### F1 Score

**F1 Score** is the harmonic mean of precision and recall, balancing both 
concerns. It penalizes models that sacrifice one for the other and is the 
standard metric for imbalanced classification tasks.

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

### Micro, Macro, and Weighted F1

Since this is a multi-label classification task with 10 article labels, three 
F1 variants are reported:

**Micro F1**: computes TP, FP, and FN globally across all labels before 
calculating F1. This favors performance on common labels.

$$\text{Micro F1} = \frac{2 \times \sum_{i=1}^{N} TP_i}{\sum_{i=1}^{N}(2 \times TP_i + FP_i + FN_i)}$$

**Macro F1**: computes F1 for each label independently then averages equally 
across all labels. This penalizes poor performance on rare labels equally.

$$\text{Macro F1} = \frac{1}{N} \sum_{i=1}^{N} F1_i$$

**Weighted F1**: computes F1 for each label weighted by its support (number 
of true instances), balancing between Micro and Macro F1.

$$\text{Weighted F1} = \frac{\sum_{i=1}^{N} w_i \times F1_i}{\sum_{i=1}^{N} w_i}$$

Where $w_i$ is the number of true instances for label $i$.

---

### Which Metric to Focus On?

**Micro F1 is the primary metric** for this project because:
- It reflects overall performance across all predictions
- It is robust to class imbalance compared to accuracy
- It is the standard metric for multi-label classification benchmarks

The gap between Micro F1 (0.6393) and Macro F1 (0.5054) in this project 
confirms that the model struggles with rare labels — exactly as predicted 
during exploratory data analysis.

---

## False Positive vs False Negative — Which Is More Harmful?

In this legal context, a **false negative** means missing a real violation, this is 
more harmful than a **false positive** which incorrectly flagging a violation.

A missed violation means a genuine human rights abuse goes undetected, 
potentially leaving victims without remedy. A false alarm, on the other hand, 
simply means a researcher spends extra time reviewing a case that turns out 
not to be a violation, a much less serious consequence.

This asymmetry suggests that in production use, the classification threshold 
should be tuned below 0.5 to prioritize recall over precision, catching more 
violations at the cost of occasional false alarms. See the Threshold section 
in Known Limitations & Future Work in the README for more details.

---

## Key Findings

| Metric | Score |
|--------|-------|
| Micro F1 | 0.6393 |
| Macro F1 | 0.5054 |
| Weighted F1 | 0.6227 |

- **Strong performers** (F1 > 0.65) : Articles 2, 3, 5, 6, and P1-1, which 
  have sufficient training examples
- **Moderate performers** (F1 0.4–0.65) : Articles 8, 10, and 11 with 
  moderate training examples
- **Poor performers** (F1 < 0.15) : Articles 9 and 14 with very few training 
  examples (41 and 141 respectively)
- **No overfitting observed** : validation loss closely tracks training loss 
  across all 3 epochs
- **Gap between Micro and Macro F1** confirms the model struggles with rare 
  labels as predicted during exploratory data analysis

