import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
data = pd.read_csv("C:/Users/HP 840 G4/Desktop/books.csv", error_bad_lines = False)
#data = pd.read_csv("C:/Users/HP 840 G4/Desktop/flow/listing.csv", encoding = 'latin-1')
#data.head()
#data.isnull().sum()

top_books = data[data['ratings_count'] > 1000000]
top_books = top_books.sort_values(by='average_rating', ascending=False).head(20)
# top_books
top_vote = data.sort_values(by='ratings_count', ascending=False).head(20)
list(set(top_books['title'].values) - set(top_vote['title'].values))
list(set(top_vote['title'].values) - set(top_books['title'].values))

new_data = data.copy()
def fun_only_author(text):
    arlen = text.split('/')
    return arlen[0]

new_data['only_author'] = new_data['authors'].apply(lambda x : fun_only_author(x))
total_rating = new_data.drop_duplicates(subset=['only_author', 'title'], keep='first')
total_rating = total_rating.groupby(by=['only_author']).agg({'average_rating': ['sum']})
total_rating.columns = ['total_rating']
total_rating.reset_index(inplace=True)
total_rating = total_rating.sort_values(by=['total_rating'], ascending=False)
#total_rating

total_book = new_data.groupby(by=['only_author']).agg({'title': ['nunique']})
total_book.columns = ['total_book']
total_book.reset_index(inplace=True)
total_book = total_book.sort_values(by=['total_book'], ascending=False)
#total_book

avg_author = pd.merge(total_book, total_rating, on='only_author', how='outer')
avg_author['average_rating'] = round(avg_author['total_rating'] / avg_author['total_book'], 2)
avg_author = avg_author[avg_author['total_book'] > 26]
avg_author = avg_author.sort_values(by=['average_rating'], ascending=False)
#avg_author

total_vote = new_data.drop_duplicates(subset=['only_author', 'title'], keep='first')
total_vote.reset_index(inplace=True)
total_vote = total_vote[['only_author', 'title', 'average_rating', 'ratings_count']]
#total_vote

C = total_vote.average_rating.mean()
m = total_vote.ratings_count.quantile(0.9)

total_vote = total_vote[total_vote['ratings_count'] >= m]

def weighted_rating(x, m=m, C=C):
    v = x['ratings_count']
    R = x['average_rating']
    return (v/(v+m) * R) + (m/(m+v) * C)

total_vote['score'] = total_vote.apply(weighted_rating, axis=1)

total_vote = total_vote.sort_values(by='score', ascending=False).head(20)

top_pages = data.sort_values(by='  num_pages', ascending=False).head(20)

title_value = data.title.unique()

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

new_data.loc[ (new_data['average_rating'] >= 0) & (new_data['average_rating'] <= 1), 'rating_between'] = "between_0_to_1"
new_data.loc[ (new_data['average_rating'] > 1) & (new_data['average_rating'] <= 2), 'rating_between'] = "between_1_to_2"
new_data.loc[ (new_data['average_rating'] > 2) & (new_data['average_rating'] <= 3), 'rating_between'] = "between_2_to_3"
new_data.loc[ (new_data['average_rating'] > 3) & (new_data['average_rating'] <= 4), 'rating_between'] = "between_3_to_4"
new_data.loc[ (new_data['average_rating'] > 4) & (new_data['average_rating'] <= 5), 'rating_between'] = "between_4_to_5"

trial = new_data[['average_rating', 'ratings_count']]
data_model = np.asarray([np.asarray(trial['average_rating']), np.asarray(trial['ratings_count'])]).T

from sklearn.cluster import KMeans

# Elbow Method

score = []
x = data_model
for cluster in range(1,41):
    kmeans = KMeans(n_clusters = cluster, init="k-means++", random_state=40)
    kmeans.fit(x)
    score.append(kmeans.inertia_)
    
rating_between_df = new_data['rating_between'].str.get_dummies(sep=",")

lang_df = new_data['language_code'].str.get_dummies(sep=",")

engine_features = pd.concat([rating_between_df, lang_df, new_data['average_rating'], new_data['ratings_count']], axis=1)

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
engine_features = min_max_scaler.fit_transform(engine_features)

from sklearn import neighbors

engine_model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')

engine_model.fit(engine_features)

dist, idlist = engine_model.kneighbors(engine_features)

def book_recommendation_engine(book_name):
    book_list_name = []
    book_id = new_data[new_data['title'] == book_name].index
    
    if any(book_id):
        out = "available"
        if out == "available":
            book_id = book_id[0]
        #     print('book_id', book_id)
            for newid in idlist[book_id]:
        #         print(newid)
                book_list_name.append(new_data.loc[newid].title)
        #         print(new_data.loc[newid].title)
    else:
        out = "not available"
        if out == "not available":
            book_list_name = print("This book is not available in the dataset")
    
    
    return book_list_name
