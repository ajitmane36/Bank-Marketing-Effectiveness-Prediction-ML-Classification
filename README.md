# Bank-Marketing-Effectiveness-Prediction-ML-Classification
This project focuses on utilizing machine learning techniques to predict the effectiveness of bank marketing campaign. Logistic Regression, Decision Tree, Random Forest, Gradient Boosting Machine, XGBoost, K Nearest Neighbor, Naive Bayes, Support Vector Machine, and Artificial Neaural Networks algorithms are used to build a model to predict whether clients will subscribe to a term deposit or not.

#### <ins>Problem Statement</ins>
     The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The marketing campaigns
     were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank
     term deposit) would be ('yes') or not ('no') subscribed. The classification goal is to predict if the client will subscribe a    
     term deposit (variable y).
#### <ins>Dataset</ins>
     The data set contained details about bank marketing campaigns.
     
     Attribute Information :
     age: age of client
     job : type of job
     marital : marital status
     education: qualification of client
     default: has credit in default? 
     balance: Account balance of client
     housing: has housing loan? (categorical: 'no','yes','unknown')
     loan: has personal loan? 
     contact: contact communication type 
     month: last contact month of year
     day: last contact day of the month
     duration: last contact duration, in seconds
     campaign: number of contacts performed during this campaign and for this client
     pdays: number of days that passed by after the client was last contacted from a previous campaign
     previous: number of contacts performed before this campaign and for this client (numeric)
     poutcome: outcome of the previous marketing campaign
     y (target Variable) - has the client subscribed a term deposit? (binary: 'yes','no')
#### <ins>Data Cleaning and Data Preprocessing</ins>
     [1] Handling Duplicate Values :
     - Dataset having no duplicated values.
     [2] Handling Null / Missing Values :
     - Null values are replaced with the mode of that respective feature, and a feature is removed if it has null values greater than 50%.
     - The null values for poutcome, contact, education, and job are 81.75%, 28.80%, 4.11%, and 0.64%, respectively.
     [3] Handling Outliers :
     - Interquartile Range in the skew symmetric curve used to remove outliers found in the the variables age, balance, duration, campaign, pdays, and previous.
     [4] Features encoding :
     - For the categorical variables marital, education, default, housing, loan, contact, and y, label encoding applied, which have 
     limited number of categories. One hot encoding used for the categorical variables job and month, which have a large number of categories.
     [5] Handling Imbalanced Dataset :
     - Class imbalance handled successfully using the Synthetic Minority Oversampling Technique (SMOTE).
     [6] Data Scaling :
     - MinMaxScaler keeps the original distribution's shape. The information present in the original data is not significantly altered.
     So, to scale the dataset, we utilised MinMaxScaler.
#### <ins>Exploratory Data Analysis</ins>
     These following graphs and plots were primarily created using Matplotlib and the Seaborn package.
     - Bar plot, count plot, pair plot, dist plot, box plot, nad heatmap
     
     Performed EDA and reached the following conclusions:
     - The average client is between the ages of 25 and 60, but the majority of bank term deposits are made by clients between the ages of 30 and 36.
     - Most clients with blue-collar jobs do not subscribe to bank term deposits (20.52%), but most clients with managerial jobs do (2.88%).
     - Most of the clients are married. Clients who are married are the most likely to subscribe to term deposits, and they are also the least likely to subscribe to term deposits.
     - Most of the clients are married. Clients who are married are the most likely to subscribe to term deposits, and divorced clients are less likely to subscribe to term deposits.
     - Clients who are more educated than the primary are more likely to sign up for a term deposit.
     - Most of the clients who subscribed to term deposits have no credit in default.
     - The majority of clients who have signed up for a term deposit do not have any housing loan.
     - If a client has a housing loan, there is a 51% chance that they will not subscribe to a term deposit.
     - Clients are more likely to subscribe to the term deposit if they do not have any personal loans.
     - If the client has a personal loan, there is a greater chance that they will not subscribe to a term deposit.
     - The clients who were contacted with celluler are mostly subscribed to term deposits.
     - Less than one percent of total clients contacted per day subscribe to term deposits.
     - In May, June, July, August, and April, more than 1 percentage of clients subscribed to the term deposit, but other than this month,
     less than 1 percentage of clients subscribed to the term deposit.
     - In June, July, August, and April, more than 1 percentage of clients subscribed to the term deposit, but other than this month,
     less than 1 percentage of clients subscribed to the term deposit. May's subscriber rate is more than double that of the other months of the
     year, a difference of more than 2 percentage.
     - No one has signed up for term deposit if they have received more than three phone calls. Less than three times contacted clients who signed up for term deposits.
     - Only 11.7% of total clients sign up for term deposits, which means that there is an 88.3% chance that clients will not subscribe to term deposits.
     - Most clients who have management-related jobs and a tertiary degree have subscribed to the term deposits.
     - Customers with a secondary education are the second most likely to subscribe to term deposits.
     - Clients are more likely to subscribe to term deposits if they spend more time on the phone.
     - Average of 400 seconds required to convey clients' intent to subscribe and make a term deposit
     - A customer is more likely to sign up for a term deposit if he is entirely debt-free.
     - Customers are less likely to choose a term deposit if they already have both types of loans.
#### <ins>Model Building</ins>
     We implemented the following machine learning models:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Gradient Boosting Machine
     - XGBoost
     - K Nearest Neighbor
     - Naive Bayes
     - Support Vector Machine
     - Artificial Neaural Networks
#### <ins>Models Evaluation</ins>
     - Models and their accuracy scores are: Logistic Regression: 0.91; Decision Tree: 0.82; Random Forest: 0.86; Gradient Boosting Machine: 0.92;
     XGBoost: 0.93 ; K Nearest Neighbor: 0.91; Naive Bayes: 0.85; Support Vector Machine: 0.91; and Artificial Neural Networks: 0.91.
     - Model XGBoost tops all classification evaluation metrics among all different implemented models.
#### <ins>Model Explainability and Feature Importance</ins>
     - The top five features are duration, age, month may, housing, and day, listed in decreasing order of their impact on a model's predictions. 
     - Overall, we can draw the conclusion that lower values of the majority of the input features have a positive impact on the model's prediction,
     but higher values of the majority of the input features have a negative impact.
#### <ins>Conclusion</ins>
     - The XGBoost classification model has the highest accuracy, precision, recall, and F1-score of all the models.
     Furthermore, XGBoost has a roc auc score of 0.93, which is very close to one, indicating that the classifier is perfectly capable of differentiating between classes.
     - The XGBoost classification model trained using cross validation is the ideal model and well-trained for predicting whether 
     the client will subscribe to a term deposit or not due to its high accuracy (0.936126), precision (0.93),
     recall (0.93), F1 score (0.93), and rou auc score (0.93), which is close to one.
