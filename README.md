# VSO Churn

# Abstract

In this churn prediction project, I attempted to predict customer retention for the Vancouver Symphony Orchestra (VSO). The VSO uses Tessitura software, which outputs data in the form of SQL tables. Using several of these tables, I found that support vector machine and logistic regression both resulted in AUC (ROC), precision, recall, and F1 score of approximately 0.67 for all error metrics. This is not ideal, but better than random, and I believe acceptable given the limited data. I used LIME for interpretability.

# Skills

This project deals with (very messy!) real world data. Completed in Python, it demonstrates my experience with data cleaning, preprocessing, and predictive analytics using machine learning using scikit-learn.

# Data

The data for this project is obtained from Tessitura, which is an arts enterprise software. Tessitura helps arts organizations manage their customer and sales information in a SQL database made up of hundreds of tables. I was granted access to 8 of these tables, that detailed customer subscription and contact information, as well a ticket sales and performance details.

# Process

We will refer to the “positive” class as those who were labelled “inactive,” and therefore churned. The “negative” class then refers to customers who were labelled in the Tessitura system to be “active” and therefore current customers.

I started by cleaning the data. The rough process can be seen in a Jupyter notebook on my github. After cleaning, there were about 1200 data points, of which only 3% were positive. This is very imbalanced, which is a common difficulty with these sorts of problems. For this exercise, I chose to under sample the negative class to rectify this, rather than oversample the positive class, since there were so few positive samples to begin with.

Next was preprocessing. I encoded the categorical variables with LabelEncoder(), then standardized (mean 0, std 1) the features, which I found to work better for my predictions than normalization (rows normalized to length 1) or scaling (each feature value in (0,1)).

I then began with predictive modelling. First, using K-folds cross validation (10 folds), I trained several naïve models: SVM, KNN, Decision Tree, and Logistic Regression. For all of these, I obtained a mean cross validation accuracy of about 55%. Not so great.

I chose to allocate 75% of the data to a training set, and left out 25% to test on. I then did grid search cross validation using the training set to determine the optimal parameters for SVM, Logistic Regression, and KNN. I used AUC (ROC), precision, recall, and F1 score as error metrics. I threw in accuracy too just to see. I also looked at a naïve decision tree classifier again.

In the end, SVM and Logistic Regression came out on top, with error metrics all about 0.67.

Finally, I used LIME as a fun little exercise, to see what features were the most important. It turns out that important features were the customer “priority code” (not sure what that is- that’s a limitation!) and the most recent solicitor they interacted with. It’s certainly a bad sign if a solicitor has an impact on whether a customer will return or not.

# Conclusion: Discussion and Limitations

In this project, I got to work with real world data: messy and limited in size. While 0.67 isn’t a great score for any of my error metrics, it’s certainly better than random, and pretty good, I think, given the limited data. If I had access to more of the Tessitura data, I would have had a better time at predicting churn. Tessitura supports hundreds of arts companies across Canada, so I think it’d be interesting to do some sort of ensemble method with the other organizations. This could help rectify the data limitation issue. In addition, the data is ridiculously messy- something I think both the VSO and Tessitura could work on. I think despite the low predictive capability of the models, the results present business value to the VSO. They could use the model as an aid to help decide if a customer is going to forego buying a ticket to a show, or a new season pass. If this is the case, they can throw a marketing package at them. The models would certainly be useful tools.

