import matplotlib
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from IPython.display import display
from sklearn import metrics





#df = pd.read_csv('Ticket N1 _ N2 - Feuil2.csv')
#df = pd.read_csv('test.csv')
df = pd.read_csv('output.csv')
df.head()

col = ['group', 'body']
df = df[col]
df = df[pd.notnull(df['body'])]
df.columns = ['group', 'body']
df['group_id'] = df['group'].factorize()[0]
group_id_df = df[['group', 'group_id']].drop_duplicates().sort_values('group_id')
group_to_id = dict(group_id_df.values)
id_to_group = dict(group_id_df[['group_id', 'group']].values)
df.head()

fig = plt.figure(figsize=(8,6))
df.groupby('group').body.count().plot.bar(ylim=0)
#plt.show()
plt.savefig('Imbalanced_Classes.png')
plt.close()


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.body).toarray()
labels = df.group_id
features.shape


N = 2
for group, group_id in sorted(group_to_id.items()):
  features_chi2 = chi2(features, labels == group_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(group))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

X_train, X_test, y_train, y_test = train_test_split(df['body'], df['group'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)


print("_____PREDICTION__-->______")
print(clf.predict(count_vect.transform([""])))

df[df['body'] == ""]
print("___<--___PREDICTION_______")

models = [
#    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC()
 #   MultinomialNB(),
  #  LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []

for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV) # Warning 
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
#plt.show()
plt.savefig('Precision.png')
plt.close()


cv_df.groupby('model_name').accuracy.mean()

model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=group_id_df.group.values, yticklabels=group_id_df.group.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
#plt.show()
plt.savefig('Prediction_accurency.png')
plt.close()


# misclassifications

for predicted in group_id_df.group_id:
  for actual in group_id_df.group_id:
    if predicted != actual and conf_mat[actual, predicted] >= 10:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_group[actual], id_to_group[predicted], conf_mat[actual, predicted]))
      display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['group', 'body']])
      print('')

model.fit(features, labels)
N = 2
for group, group_id in sorted(group_to_id.items()):
  indices = np.argsort(model.coef_[group_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  print("# '{}':".format(group))
  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))

print(metrics.classification_report(y_test, y_pred, target_names=df['group'].unique()))









