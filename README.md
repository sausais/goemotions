# MD2 Valodu tehnoloģiju pamatjēdzieni

## Text preprocessing

Text preprocessing is used to process the test.tsv and train.tsv for training
```bash
python3 text_preprocessing.py
```
Emotions are mapped with ekman_mapping.json and emotions.txt

Multiple techinques are using for preprocessing data such as:
- Normalization
- Tokenization
- Trimming


Training is done with Naive Bayes classification method

Training is also done with pretrained model bert-base-uncased which was fine-tuned.

Inference can be run with 
```bash
python3 inference.py
```


## Results:
### nb_trimmed:

              precision    recall  f1-score   support

       anger       0.36      0.42      0.38       785
     disgust       0.13      0.48      0.21       123
        fear       0.16      0.62      0.25       101
         joy       0.78      0.56      0.65      2404
     neutral       0.50      0.33      0.40      1787
     sadness       0.29      0.53      0.37       406
    surprise       0.34      0.43      0.38       723

    accuracy                           0.46      6329
![alt text](/assets/nb_trimmed.png)

### nb_tokenized:

              precision    recall  f1-score   support

       anger       0.36      0.43      0.39       785
     disgust       0.14      0.49      0.22       123
        fear       0.16      0.60      0.26       101
         joy       0.77      0.58      0.66      2404
     neutral       0.52      0.30      0.38      1787
     sadness       0.30      0.54      0.39       406
    surprise       0.33      0.46      0.38       723

    accuracy                           0.47      6329
![alt text](/assets/nb_tokenized.png)
### nb_normalized:

              precision    recall  f1-score   support

       anger       0.34      0.39      0.36       785
     disgust       0.13      0.47      0.20       123
        fear       0.16      0.64      0.25       101
         joy       0.75      0.58      0.65      2404
     neutral       0.48      0.29      0.36      1787
     sadness       0.30      0.51      0.38       406
    surprise       0.29      0.37      0.33       723

    accuracy                           0.45      6329
![alt text](/assets/nb_normalized.png)


### Results Bert fine tune
The fine-tuning was done on A100 2 epochs 32 batch-size

              precision    recall  f1-score   support

       anger       0.56      0.55      0.55       785
     disgust       0.52      0.37      0.43       123
        fear       0.60      0.62      0.61       101
         joy       0.79      0.85      0.82      2404
     neutral       0.65      0.61      0.63      1787
     sadness       0.63      0.56      0.59       406
    surprise       0.57      0.58      0.58       723

    accuracy       0.61      0.59      0.685      6329
    macro avg     0.619      0.59      0.604      6329

![alt text](/assets/fine_tune.png)
