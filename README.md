# Comparing Classifiers

### Overview: ###

In this practical application, your goal is to compare the performance of the classifiers we encountered in this section, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines. We will utilize a dataset related to marketing bank products over the telephone. 
 
### Data: ###

The data can be found in this link

[bank-additional-full.csv](https://github.com/camorante/Comparing-Classifiers/tree/main/data/bank-additional-full.csv)

### Notebook Link ###

The following notebook contains all the development of the analysis carried out.

[prompt_III.ipynb](https://github.com/camorante/Comparing-Classifiers/blob/main/prompt_III.ipynb)

### Business Understanding ###

The main task to be performed in this particular case is to understand what makes a customer accept or not to subscribe to a bank account (deposit) and to verify which are the characteristics that most influence this decision. For this purpose, a dataset of 41187 records has been provided from which information has been collected on several characteristics related to the customer.

### Data Understanding ###

Examine the data description below, and determine if any of the features are missing values or need to be coerced to a different data type.

Input variables:

Bank client data:  <br />
1 **age** (numeric)  <br />
2 **job** : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')  <br />
3 **marital** : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)  <br />
4 **education**: (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')  <br />
5 **default**: has credit in default? (categorical: 'no','yes','unknown')  <br />
6 **housing**: has housing loan? (categorical: 'no','yes','unknown')  <br />
7 **loan**: has personal loan? (categorical: 'no','yes','unknown')  <br />

Related with the last contact of the current campaign:  <br />
8 **contact**: contact communication type (categorical: 'cellular','telephone')  <br />
9 **month**: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')  <br />
10 **day_of_week**: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')  <br />
11 **duration**: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.  <br />

Other attributes:  <br />
12 **campaign**: number of contacts performed during this campaign and for this client (numeric, includes last contact)  <br />
13 **pdays**: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)  <br />
14 **previous**: number of contacts performed before this campaign and for this client (numeric)  <br />
15 **poutcome**: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')  <br />

Social and economic context attributes:  <br />
16 **emp.var.rate**: employment variation rate - quarterly indicator (numeric)  <br />
17 **cons.price.idx**: consumer price index - monthly indicator (numeric)  <br />
18 **cons.conf.idx**: consumer confidence index - monthly indicator (numeric)  <br />
19 **euribor3m**: euribor 3 month rate - daily indicator (numeric)  <br />
20 **nr.employed**: number of employees - quarterly indicator (numeric)  <br />

Output variable (desired target):  <br />
21 **y**: has the client subscribed a term deposit? (binary: 'yes','no')  <br />


### Data Preparation and Cleaning ###

In this dataset there are no null values to fix and no imputations have been performed.

#### Correlations and Imbalance: ####

<ins>Correlations HeatMap</ins>

![](images/fig1.png)

There are several characteristics with a high correlation ('emp.var.rate', 'cons.price.idx', 'nr.employed')

<ins>Imbalance</ins>

![](images/fig2.png)

As can be seen, the target variable is very unbalanced, which may affect the predictions.

### Baseline Model ###

For the training I used 4 different models in order to find the most optimal one (A dummy classifier will be used as a baseline reference model):

Only 7 features were used

* Logistic Regression
* KNN
* Decision Tree
* SVM

I use the default settings for each of the models

### Baseline Model Evaluation ###

![](images/fig3.png)

![](images/fig4.png)

These results show that the best method for this specific dataset is Decision Tree since it has a good training time and in addition to a good accuracy(not the highest but close) it also has the highest F1 score. These models using only 7 characteristics are not ideal for prediction, they are too close to the base model (Dummy Classifier), it is notorious that it is necessary to add more characteristics to them.