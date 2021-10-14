
# import librairies for the code to work or just use google colab.
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as math
import time
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [14, 14]

# load the data
df = pd.read_csv('./sample_data/ANIME.csv')

df['categories'] = df['genre'].apply(lambda l: [] if pd.isna(l) else [
                                     i.strip() for i in l.split(",")])


df.head()


# Build the tfidf matrix with the descriptions
start_time = time.time()
text_content = df['synopsis']
vector = TfidfVectorizer(max_df=0.4,         # drop words that occur in more than X percent of documents
                         min_df=1,      # only use words that appear at least X times
                         stop_words='english',  # remove stop words
                         lowercase=True,  # Convert everything to lower case
                         use_idf=True,   # Use idf
                         norm=u'l2',     # Normalization
                         smooth_idf=True  # Prevents divide-by-zero errors
                         )
df.synopsis = df.synopsis.fillna(' ')
tfidf = vector.fit_transform(text_content)

# Clustering  Kmeans
k = 200
kmeans = MiniBatchKMeans(n_clusters=k)
kmeans.fit(tfidf)
centers = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vector.get_feature_names()

request_transform = vector.transform(df['synopsis'])
# new column cluster based on the description
df['cluster'] = kmeans.predict(request_transform)

# Find similar : get the top_n movies with description similar to the target description


def find_similar(tfidf_matrix, index, top_n=5):
    cosine_similarities = linear_kernel(
        tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[
        ::-1] if i != index]
    return [index for index in related_docs_indices][0:top_n]


G = nx.Graph(label="ANIME")
start_time = time.time()
for i, rowi in df.iterrows():
    if (i % 1000 == 0):
        print(" iter {} -- {} seconds --".format(i, time.time() - start_time))
    # mtype=rowi['type'],rating=rowi['rating'])
    G.add_node(rowi['title'], key=rowi['uid'],
               label="ANIME", ranking=rowi['ranked'])
    #G.add_edge(rowi['title'], element, label="ACTED_IN")
    for element in rowi['genre']:
        G.add_node(element, label="G")
        G.add_edge(rowi['title'], element, label="GEN_IN")

    indices = find_similar(tfidf, i, top_n=5)
    snode = "Sim("+rowi['title'][:15].strip()+")"
    G.add_node(snode, label="SIMILAR")
    G.add_edge(rowi['title'], snode, label="SIMILARITY")
    for element in indices:
        G.add_edge(snode, df['title'].loc[element], label="SIMILARITY")
print(" finish -- {} seconds --".format(time.time() - start_time))


def get_all_adj_nodes(list_in):
    sub_graph = set()
    for m in list_in:
        sub_graph.add(m)
        for e in G.neighbors(m):
            sub_graph.add(e)
    return list(sub_graph)


def draw_sub_graph(sub_graph):
    subgraph = G.subgraph(sub_graph)
    colors = []
    for e in subgraph.nodes():
        if G.nodes[e]['label'] == "ANIME":
            colors.append('blue')
        elif G.nodes[e]['label'] == "G":
            colors.append('green')
        elif G.nodes[e]['label'] == "SIMILAR":
            colors.append('orange')
        elif G.nodes[e]['label'] == "CLUSTER":
            colors.append('orange')

    nx.draw(subgraph, with_labels=True, font_weight='bold', node_color=colors)
    plt.show()


list_in = ["Kimetsu no Yaiba", "Yakusoku no Neverland"]
sub_graph = get_all_adj_nodes(list_in)
draw_sub_graph(sub_graph)


def get_recommendation(root):
    commons_dict = {}
    for e in G.neighbors(root):
        for e2 in G.neighbors(e):
            if e2 == root:
                continue
            if G.nodes[e2]['label'] == "ANIME":
                commons = commons_dict.get(e2)
                if commons == None:
                    commons_dict.update({e2: [e]})
                else:
                    commons.append(e)
                    commons_dict.update({e2: commons})
    movies = []
    weight = []
    for key, values in commons_dict.items():
        w = 0.0
        for e in values:
            w = w+1/math.log(G.degree(e))
        movies.append(key)
        weight.append(w)

    result = pd.Series(data=np.array(weight), index=movies)
    result.sort_values(inplace=True, ascending=False)
    return result


result = get_recommendation("Kimetsu no Yaiba")
result2 = get_recommendation("Yakusoku no Neverland")

print("*"*40+"\n Recommendation for 'Kimetsu no Yaiba'\n"+"*"*40)
print(result.head())
print("*"*40+"\n Recommendation for 'Yakusoku no Neverland'\n"+"*"*40)
print(result2.head())

reco = list(result.index[:4].values)
reco.extend(["Yakusoku no Neverland"])
sub_graph = get_all_adj_nodes(reco)
draw_sub_graph(sub_graph)
