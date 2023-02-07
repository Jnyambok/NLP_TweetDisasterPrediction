---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.6
  nbformat: 4
  nbformat_minor: 2
  orig_nbformat: 4
  vscode:
    interpreter:
      hash: b671c20432fcd147198c92e7f072af9e705f087eb990bee22b07f08caab9f630
---

::: {.cell .markdown}
**Natural Language Processing with Disaster Tweets**
:::

::: {.cell .markdown}
The following code is for the
`<a href ="https://www.kaggle.com/competitions/nlp-getting-started/data?select=sample_submission.csv">`{=html}
Kaggle NLP prediction `</a>`{=html} that I particpated in.

The task: You are predicting whether a given tweet is about a real
disaster or not. If so, predict a 1. If not, predict a 0 `<br>`{=html}
The files involved were: `<ul>`{=html} `<li>`{=html}- Train.csv \[The
training dataset\]`</li>`{=html} `<li>`{=html}- Test.csv \[The test
dataset\]`</li>`{=html} `<li>`{=html}- Submission.csv \[The submission
dataset\]`</li>`{=html} `</ul>`{=html}
:::

::: {.cell .markdown}
Loading the necessary libraries and the datasets
:::

::: {.cell .code execution_count="56"}
``` python
import pandas as pd    # data processing
import numpy as np   #linear algebra
import nltk
import matplotlib.pyplot as plt
from sklearn import feature_extraction, linear_model,model_selection, preprocessing #Machine learning models
```
:::

::: {.cell .code execution_count="57"}
``` python
#loading the train dataset 
train_dis = pd.read_csv('C:\\Users\\hp\\Desktop\\Programming\\Machine learning\\Data Science\\Py Notebooks\\Tragedy_sentiment_analysis\\train.csv')
train_dis.head()
```

::: {.output .execute_result execution_count="57"}
```{=html}
<div><div id=06850890-4770-4d57-9d96-d5b3a3ea7dfe style="display:none; background-color:#9D6CFF; color:white; width:200px; height:30px; padding-left:5px; border-radius:4px; flex-direction:row; justify-content:space-around; align-items:center;" onmouseover="this.style.backgroundColor='#BA9BF8'" onmouseout="this.style.backgroundColor='#9D6CFF'" onclick="window.commands?.execute('create-mitosheet-from-dataframe-output');">See Full Dataframe in Mito</div> <script> if (window.commands?.hasCommand('create-mitosheet-from-dataframe-output')) document.getElementById('06850890-4770-4d57-9d96-d5b3a3ea7dfe').style.display = 'flex' </script> <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation orders in California</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as smoke from #wildfires pours into a school</td>
      <td>1</td>
    </tr>
  </tbody>
</table></div>
```
:::
:::

::: {.cell .code execution_count="58"}
``` python
#loading the test dataset
test_dis = pd.read_csv('C:\\Users\\hp\\Desktop\\Programming\\Machine learning\\Data Science\\Py Notebooks\\Tragedy_sentiment_analysis\\train.csv')
test_dis.head()
```

::: {.output .execute_result execution_count="58"}
```{=html}
<div><div id=6c718b7f-84ce-4e8e-8aaf-b5e76d71cb66 style="display:none; background-color:#9D6CFF; color:white; width:200px; height:30px; padding-left:5px; border-radius:4px; flex-direction:row; justify-content:space-around; align-items:center;" onmouseover="this.style.backgroundColor='#BA9BF8'" onmouseout="this.style.backgroundColor='#9D6CFF'" onclick="window.commands?.execute('create-mitosheet-from-dataframe-output');">See Full Dataframe in Mito</div> <script> if (window.commands?.hasCommand('create-mitosheet-from-dataframe-output')) document.getElementById('6c718b7f-84ce-4e8e-8aaf-b5e76d71cb66').style.display = 'flex' </script> <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation orders in California</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as smoke from #wildfires pours into a school</td>
      <td>1</td>
    </tr>
  </tbody>
</table></div>
```
:::
:::

::: {.cell .code execution_count="59"}
``` python
#Checking for nulls 
train_dis.isnull().sum()
```

::: {.output .execute_result execution_count="59"}
    id             0
    keyword       61
    location    2533
    text           0
    target         0
    dtype: int64
:::
:::

::: {.cell .markdown}
The columns missing data are keyword and location `<br>`{=html} The
Keyword column shows the \"word\" from the tweets that stood out the
most such as \"He set her house ablaze\" `<br>`{=html} The Keyword will
be \"Ablaze\"

Location column shows where the tweet come from
:::

::: {.cell .markdown}
**Data Cleaning and Exploratory Data Analysis**
:::

::: {.cell .code execution_count="60"}
``` python
#Sample of what a positive tweet and negative tweet looks like
positive=train_dis[train_dis['target']==0]["text"].values[1]
negative=train_dis[train_dis['target']==1]["text"].values[1]


print('This is a positive tweet:\n',positive ,'\nThis is a negative tweet:\n',negative)
```

::: {.output .stream .stdout}
    This is a positive tweet:
     I love fruits 
    This is a negative tweet:
     Forest fire near La Ronge Sask. Canada
:::
:::

::: {.cell .markdown}
I combined the keyword column and the text column to come up with the
tweet column for the train and test data set. This adds more weight to
the tweet column for the values that don\'t have a blank keyword rows
:::

::: {.cell .code execution_count="61"}
``` python

# combine columns A and B with a space in between, if A is not null, otherwise keep B unchanged
train_dis['tweet'] = np.where(train_dis['text'].notnull(), train_dis['text'].astype(str) + ' ' + train_dis['keyword'].astype(str), train_dis['keyword'])
train_dis['tweet'] = train_dis['tweet'].str.replace('nan','')

#Viewing rows where the keywords are not null
df_not_null = train_dis[train_dis['keyword'].notnull()]

df_not_null[['keyword' ,'tweet']]
```

::: {.output .execute_result execution_count="61"}
```{=html}
<div><div id=e52f9306-d6a3-42d0-8e71-6958aadb0c20 style="display:none; background-color:#9D6CFF; color:white; width:200px; height:30px; padding-left:5px; border-radius:4px; flex-direction:row; justify-content:space-around; align-items:center;" onmouseover="this.style.backgroundColor='#BA9BF8'" onmouseout="this.style.backgroundColor='#9D6CFF'" onclick="window.commands?.execute('create-mitosheet-from-dataframe-output');">See Full Dataframe in Mito</div> <script> if (window.commands?.hasCommand('create-mitosheet-from-dataframe-output')) document.getElementById('e52f9306-d6a3-42d0-8e71-6958aadb0c20').style.display = 'flex' </script> <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>keyword</th>
      <th>tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>31</th>
      <td>ablaze</td>
      <td>@bbcmtd Wholesale Markets ablaze http://t.co/lHYXEOHY6C ablaze</td>
    </tr>
    <tr>
      <th>32</th>
      <td>ablaze</td>
      <td>We always try to bring the heavy. #metal #RT http://t.co/YAo1e0xngw ablaze</td>
    </tr>
    <tr>
      <th>33</th>
      <td>ablaze</td>
      <td>#AFRICANBAZE: Breaking news:Nigeria flag set ablaze in Aba. http://t.co/2nndBGwyEi ablaze</td>
    </tr>
    <tr>
      <th>34</th>
      <td>ablaze</td>
      <td>Crying out for more! Set me ablaze ablaze</td>
    </tr>
    <tr>
      <th>35</th>
      <td>ablaze</td>
      <td>On plus side LOOK AT THE SKY LAST NIGHT IT WAS ABLAZE http://t.co/qqsmshaJ3N ablaze</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7578</th>
      <td>wrecked</td>
      <td>@jt_ruff23 @cameronhacker and I wrecked you both wrecked</td>
    </tr>
    <tr>
      <th>7579</th>
      <td>wrecked</td>
      <td>Three days off from work and they've pretty much all been wrecked hahaha shoutout to my family for that one wrecked</td>
    </tr>
    <tr>
      <th>7580</th>
      <td>wrecked</td>
      <td>#FX #forex #trading Cramer: Iger's 3 words that wrecked Disney's stock http://t.co/7enNulLKzM wrecked</td>
    </tr>
    <tr>
      <th>7581</th>
      <td>wrecked</td>
      <td>@engineshed Great atmosphere at the British Lion gig tonight. Hearing is wrecked. http://t.co/oMNBAtJEAO wrecked</td>
    </tr>
    <tr>
      <th>7582</th>
      <td>wrecked</td>
      <td>Cramer: Iger's 3 words that wrecked Disney's stock - CNBC http://t.co/N6RBnHMTD4 wrecked</td>
    </tr>
  </tbody>
</table></div>
```
:::
:::

::: {.cell .code execution_count="62"}
``` python
#We do the same for the test data since we are using the same logic
test_dis['tweet'] = np.where(test_dis['text'].notnull(), test_dis['text'].astype(str) + ' ' + test_dis['keyword'].astype(str), test_dis['keyword'])
test_dis['tweet'] = test_dis['tweet'].str.replace('nan','')
```
:::

::: {.cell .code execution_count="63"}
``` python
#Lets go back to our dataset and drop the text column
train_dis.drop('text',axis=1,inplace=True)
train_dis.head(5)
```

::: {.output .execute_result execution_count="63"}
```{=html}
<div><div id=d680bb58-810f-4fe4-97b1-ca5c5d965437 style="display:none; background-color:#9D6CFF; color:white; width:200px; height:30px; padding-left:5px; border-radius:4px; flex-direction:row; justify-content:space-around; align-items:center;" onmouseover="this.style.backgroundColor='#BA9BF8'" onmouseout="this.style.backgroundColor='#9D6CFF'" onclick="window.commands?.execute('create-mitosheet-from-dataframe-output');">See Full Dataframe in Mito</div> <script> if (window.commands?.hasCommand('create-mitosheet-from-dataframe-output')) document.getElementById('d680bb58-810f-4fe4-97b1-ca5c5d965437').style.display = 'flex' </script> <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>target</th>
      <th>tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>13,000 people receive #wildfires evacuation orders in California</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>Just got sent this photo from Ruby #Alaska as smoke from #wildfires pours into a school</td>
    </tr>
  </tbody>
</table></div>
```
:::
:::

::: {.cell .markdown}
**Exploratory Data Analysis**
:::

::: {.cell .markdown}
A wordcloud provides the visual perspective needed to get a sense of the
words that were used the most.
:::

::: {.cell .code execution_count="77"}
``` python
from wordcloud import WordCloud

#replace empty values with an empty string
train_dis['keyword'].fillna('',inplace=True)


#concatenate all text in the column into one string
text = ' '.join(train_dis['keyword'].tolist())

#create a wordcloud from the text
wordcloud = WordCloud().generate(text)

#plot the wordcloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```

::: {.output .display_data}
![](vertopal_3190e16055c1495785ea62c5235a8d0b/2e01e38af6a69aa9e8931869c0a1ce23fffb1ebd.png)
:::
:::

::: {.cell .markdown}
**Data Preprocessing**
:::

::: {.cell .markdown}
We first begin by turning the column into data that the model can
process. This can be done through COUNT VECTORIZER which counts the
words and turns them into a vector which is a set of numbers that an ML
model can work with
:::

::: {.cell .code execution_count="65"}
``` python
count_vectorizer = feature_extraction.text.CountVectorizer()

#Counts for the first 5 tweets
example_train = count_vectorizer.fit_transform(train_dis["tweet"][0:5])
example_train
```

::: {.output .execute_result execution_count="65"}
    <5x54 sparse matrix of type '<class 'numpy.int64'>'
    	with 61 stored elements in Compressed Sparse Row format>
:::
:::

::: {.cell .markdown}
The .todense() method is used to convert a sparse matrix to a dense
matrix. A sparse matrix is a matrix where most of the elements are zero.
To store this matrix efficiently, it is stored in a sparse format, which
only stores the non-zero elements.

The .todense() method converts this sparse representation of the matrix
to a dense matrix, which stores all elements including the zeros. This
conversion is useful when you want to perform computations or operations
that require the full matrix representation, or when you want to
visualize the matrix.

However, converting a sparse matrix to a dense matrix can be
memory-intensive, especially if the matrix is large and has a high
number of non-zero elements. So, it\'s recommended to use sparse
matrices wherever possible, and only use .todense() when necessary.
:::

::: {.cell .code execution_count="66"}
``` python
print(example_train[0].todense().shape)
print(example_train[0].todense())
```

::: {.output .stream .stdout}
    (1, 54)
    [[0 0 0 1 1 1 0 0 0 0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 1 0
      0 0 0 1 0 0 0 0 0 0 0 0 0 1 1 0 1 0]]
:::
:::

::: {.cell .markdown}
This means that there is 54 unique words( or \"tokens\") in the first 5
tweets and the first tweet contains only some of those unique tokens -
all the 1\'s above are the tokens that exist in the first tweet
:::

::: {.cell .code execution_count="67"}
``` python
#Creating the vectors for the tweets
train_vectors = count_vectorizer.fit_transform(train_dis["tweet"])
```
:::

::: {.cell .markdown}
We are not using .fit_transform() because using .transform() makes sure
that the tokens in the train vectors are the only ones mapped to the
test vectors : the train and test vectors use the same set of tokens

Also the test data just needs to be transformed not fit
:::

::: {.cell .code execution_count="68"}
``` python
test_vectors = count_vectorizer.transform(test_dis['tweet'])
```
:::

::: {.cell .markdown}
**Model Building and Evaluation through Cross Validation**
:::

::: {.cell .markdown}
Our vectors are big and pushing our weights toward 0 without completely
discounting different words thus using Ridge regression would be
advisable
:::

::: {.cell .code execution_count="69"}
``` python
clf = linear_model.RidgeClassifier()
```
:::

::: {.cell .code execution_count="70"}
``` python
#Using Cross Validation to see how our data would perform on unseen (the test data)

scores = model_selection.cross_val_score( clf, train_vectors, train_dis['target']
,cv = 10, scoring = "f1")
scores
```

::: {.output .execute_result execution_count="70"}
    array([0.55909944, 0.36681223, 0.39308176, 0.35343619, 0.4772118 ,
           0.48358209, 0.46231156, 0.42176871, 0.57676903, 0.63433814])
:::
:::

::: {.cell .markdown}
The return score is F1, a combonation of precision call on each of the
folds created during cross validation specified by \'cv\'. The model
isnt that bad as it would achieve atleast 0.63 Theres room for
improvement using better models such as the Logistic Regression using
TFIFD as the vectorizer

The TFIFD model is a numerical statistic used to evaluate how important
a word is to a document in a collection or corpus. It is often used as a
weighting factor in information retrieval and text mining.

The basic idea behind TF-IDF is that words that appear frequently in a
document are important, but words that appear frequently in many
documents across the corpus are not as important. The \"TF\" component
represents the term frequency, or the number of times a word appears in
a document. The \"IDF\" component represents the inverse document
frequency, or the number of documents in the corpus divided by the
number of documents containing the word.

The final TF-IDF score for a word in a document is the product of its
term frequency and inverse document frequency. This score can then be
used to rank the importance of words in a document and to determine the
most relevant words for a given query.

We will switch to a TFIDF model to see how the score improves
:::

::: {.cell .markdown}
**Using TFIDF vectorizer with Logistic Regression model**
:::

::: {.cell .code execution_count="78"}
``` python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score

#The input data
corpus = train_dis["tweet"]
labels = test_dis["target"]

# Split the data into training and test sets
X_train,X_test,y_train,y_test = train_test_split(corpus,labels,test_size=0.2, random_state=42)


# Fit the TF-IDF vectorizer on the training data
vectorizer = TfidfVectorizer()
X_train_vectorizer=vectorizer.fit_transform(X_train)

#Train a logistic Regression model on the vectorized training data
classifier = LogisticRegression()
classifier.fit(X_train_vectorizer, y_train)

#Vectorize the test data
X_test_vectorized = vectorizer.transform(X_test)

#Predict the labels
y_pred = classifier.predict(X_test_vectorized)

#Calculate the accuracy
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

#Calculate the F1 score
f1 = f1_score(y_test,y_pred, average = 'weighted')
print("F1 score:", f1)



```

::: {.output .stream .stdout}
    Accuracy: 0.7977675640183848
    F1 score: 0.7964083945530092
:::
:::

::: {.cell .markdown}
The accuracy and F1 score has significantly improved! Logistic
regression seems to have an accuracy of 80% meaning that if the model
was to classify positive tweets, more times than not, 80% will be
correctly positive tweets and vice versa
:::

::: {.cell .code execution_count="73"}
``` python
#finishing the test
submission = pd.read_csv('C:\\Users\\hp\\Desktop\\Programming\\Machine learning\\Data Science\\Py Notebooks\\Tragedy_sentiment_analysis\\ignore\\sample_submission.csv')
submission.head()
```

::: {.output .execute_result execution_count="73"}
```{=html}
<div><div id=b4dbf14b-c918-4e28-90f4-e39192eee74d style="display:none; background-color:#9D6CFF; color:white; width:200px; height:30px; padding-left:5px; border-radius:4px; flex-direction:row; justify-content:space-around; align-items:center;" onmouseover="this.style.backgroundColor='#BA9BF8'" onmouseout="this.style.backgroundColor='#9D6CFF'" onclick="window.commands?.execute('create-mitosheet-from-dataframe-output');">See Full Dataframe in Mito</div> <script> if (window.commands?.hasCommand('create-mitosheet-from-dataframe-output')) document.getElementById('b4dbf14b-c918-4e28-90f4-e39192eee74d').style.display = 'flex' </script> <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>0</td>
    </tr>
  </tbody>
</table></div>
```
:::
:::

::: {.cell .code execution_count="74"}
``` python
#Converting the prediction dataset into a dataframe
pred = pd.DataFrame({'target': y_pred})
print(len(pred))

#Appending the predictions
submission['target'] = pred
submission.head()

```

::: {.output .stream .stdout}
    1523
:::

::: {.output .execute_result execution_count="74"}
```{=html}
<div><div id=78242d1e-2c81-4868-9c5d-3fa4d6ccb941 style="display:none; background-color:#9D6CFF; color:white; width:200px; height:30px; padding-left:5px; border-radius:4px; flex-direction:row; justify-content:space-around; align-items:center;" onmouseover="this.style.backgroundColor='#BA9BF8'" onmouseout="this.style.backgroundColor='#9D6CFF'" onclick="window.commands?.execute('create-mitosheet-from-dataframe-output');">See Full Dataframe in Mito</div> <script> if (window.commands?.hasCommand('create-mitosheet-from-dataframe-output')) document.getElementById('78242d1e-2c81-4868-9c5d-3fa4d6ccb941').style.display = 'flex' </script> <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table></div>
```
:::
:::

::: {.cell .code execution_count="75"}
``` python
#Extract the file
submission.to_csv("sample_submission.csv",index="False")
```
:::
