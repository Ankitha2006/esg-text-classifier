KenyaESG, Apache-2.0 license
Classifies sustainability report sentences into Environmental, Social and Governance categories using TF-IDF and Logistic Regression.
=== Environmental ===
  columns found: ['Unnamed: 0', 'text', 'env', 'source']
  using text='text', label='env', rows=3900
  positive (yes) share: 41.7%
              precision    recall  f1-score   support

          No       0.94      0.98      0.96       455
         Yes       0.96      0.91      0.94       325

    accuracy                           0.95       780
   macro avg       0.95      0.94      0.95       780
weighted avg       0.95      0.95      0.95       780


=== Social ===
  columns found: ['Unnamed: 0', 'text', 'soc', 'source']
  using text='text', label='soc', rows=3900
  positive (yes) share: 45.0%
              precision    recall  f1-score   support

          No       0.93      0.95      0.94       429
         Yes       0.93      0.91      0.92       351

    accuracy                           0.93       780
   macro avg       0.93      0.93      0.93       780
weighted avg       0.93      0.93      0.93       780


=== Governance ===
  columns found: ['Unnamed: 0', 'text', 'gov', 'source']
  using text='text', label='gov', rows=3900
  positive (yes) share: 38.8%
              precision    recall  f1-score   support

          No       0.92      0.95      0.93       478
         Yes       0.91      0.88      0.89       302

    accuracy                           0.92       780
   macro avg       0.92      0.91      0.91       780
weighted avg       0.92      0.92      0.92       780


=== SUMMARY (use these numbers on your CV) ===
       Pillar  Train rows  Test rows  Accuracy  F1 (Yes class)  Macro F1
Environmental        3120        780     0.947           0.935     0.945
       Social        3120        780     0.929           0.921     0.929
   Governance        3120        780     0.919           0.894     0.914

[ ]
