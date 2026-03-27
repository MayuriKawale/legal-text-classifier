# Legal Text Classifier: Executive Summary

## What Is This Project?

Every year, thousands of human rights cases are decided by the European Court 
of Human Rights. Each case contains detailed facts describing what happened to 
the applicant and which of their rights were violated. Reviewing this volume 
of text manually is slow, expensive, and requires significant legal expertise.

This project builds a tool that automates that process. Given the facts of a 
case, it predicts which human rights articles were violated in seconds, with 
no legal expertise required to operate.

---

## How Does It Work?

A state-of-the-art language model (DistilBERT) was trained on 11,000 real 
court cases from the European Court of Human Rights. The violation labels 
were determined by the court itself, not by human annotators, making them 
highly reliable.

The model learns patterns in legal language that are associated with specific 
types of violations and applies that knowledge to new, unseen cases.

---

## Who Can Use This?

- **Legal researchers** looking to screen large volumes of case text quickly
- **Human rights advocates** tracking patterns of violations across cases
- **Policy analysts** identifying which rights are most frequently at risk
- **Court administrators** flagging cases for priority review

---

## Results at a Glance

Tested on 1,000 cases the model had never seen during training:

| Metric | Score |
|--------|-------|
| Micro F1 | 0.6393 |
| Macro F1 | 0.5054 |
| Weighted F1 | 0.6227 |

The model performs well for frequently occurring violation types and 
struggles with rare ones, a known challenge when training data is limited 
for certain categories. No overfitting was observed during training.

---

## Key Strengths

- **Fast:** predictions generated in seconds per case
- **Accessible:** accepts plain text input, no legal expertise needed
- **Transparent:** outputs probability scores for each article, so users 
  can adjust sensitivity based on their needs
- **Reliable:** fully tested codebase with automated tests, version 
  control, and documented pipelines

---

## Limitations and Next Steps

The model works best with detailed case facts of 100 or more words. 
Performance on rarely occurring violation types is limited by available 
training data. Planned improvements include:

- Using a more powerful language model capable of processing longer documents
- Applying techniques to better handle rare violation categories
- Tuning the sensitivity threshold to prioritize catching violations over 
  avoiding false alarms, which is critical in a legal context

---

## Learn More

Full project including code, documentation, and technical evaluation:
[github.com/MayuriKawale/legal-text-classifier](https://github.com/MayuriKawale/legal-text-classifier)

For technical details on evaluation metrics and methodology, see 
[TECHNICAL_NOTES.md](TECHNICAL_NOTES.md).