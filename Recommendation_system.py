
# import librairies for the code to work or just use google colab.
import tkinter as tk
from tkinter import messagebox
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as math
import time

#tkinter window
root=tk.Tk() 
root.title("Recommendations")
root.state("zoomed")
title_frame = tk.Frame(root, bg='#FAEBD7', bd=4)
title_frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.1, anchor='n')
lower_frame = tk.Frame(root, bg='#FAEBD7', bd=15)
lower_frame.place(relx=0.5, rely=0.25, relwidth=0.75, relheight=0.6, anchor='n')
##background_image = tk.PhotoImage(file="Bg1.png")
##bgimage_label = tk.Label(root, image=background_image)
##bgimage_label.place(relwidth=1, relheight=1)

def main():
    def forget_grid():
        title_grid = title_frame.grid_slaves()
        lower_grid = lower_frame.grid_slaves()
        for i in title_grid:
            i.grid_forget()
        for i in lower_grid:
            i.grid_forget()
    def forget_pack():  
        title_pack = title_frame.pack_slaves()
        lower_pack = lower_frame.pack_slaves()
        for i in title_pack:
            i.pack_forget()
        for i in lower_pack:
            i.lower_forget()
            
    main_title = tk.Label(title_frame, text="RECOMMENDATION SYSTEM", bg='#FAEBD7', font=("Helvetica", 16))
    main_title.pack()
    
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
##            try:
##                t=" iter {} -- {} seconds --".format(i, time.time() - start_time)
##                count = tk.Label(lower_frame, text= t, bg='#FAEBD7', font=("Arial", 12))
##                count.grid(column=1, row=i)
##            except:
##                forget_grid()
##                forget_pack()
##                t=" iter {} -- {} seconds --".format(i, time.time() - start_time)
##                count = tk.Label(lower_frame, text= t, bg='#FAEBD7', font=("Arial", 12))
##                count.grid(column=1, row=i)
                
            
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
##    t=" finish -- {} seconds --".format(time.time() - start_time)
##    last = tk.Label(lower_frame, text= t, bg='#FAEBD7', font=("Arial", 12))
##    last.grid(column=1, row=i)

    def destroy():
        root.destroy()
    exit_button = tk.Button(lower_frame, text="Exit", font=("Helvetica", 12), command = destroy)
    exit_button.grid(column=2, row=13)


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

    def check():
        try:
            global result
            if anime.get()=="Kimetsu no Yaiba":
                result= get_recommendation("Kimetsu no Yaiba")
            if anime.get()=="Yakusoku no Neverland":
                result= get_recommendation("Yakusoku no Neverland")

            #print("*"*40+"\n Recommendation for '{}'\n".format(anime)+"*"*40)
            #print(result.head())
            forget_pack()
            forget_grid()
            main_title = tk.Label(title_frame, text="RECOMMENDED FOR YOU!!", bg='#FAEBD7', font=("Helvetica", 16))
            main_title.pack()
            t1= "*"*40+"\n Recommendation for '{}'\n".format(anime)+"*"*40
            t2= result.head()
            final1 = tk.Label(lower_frame, text=t1, bg='#FAEBD7', font=("Helvetica", 12))
            final1.pack()
            final2 = tk.Label(lower_frame, text=t2, bg='#FAEBD7', font=("Helvetica", 12))
            final2.pack()

            #On the terminal
            reco = list(result.index[:4].values)
            reco.extend(["Yakusoku no Neverland"])
            sub_graph = get_all_adj_nodes(reco)
            draw_sub_graph(sub_graph)
        except:
             messagebox.showerror(title="Invalid Entry", message="Please enter a valid name")   

    #Entries
    entry_label= tk.Label(lower_frame, text="Enter your favourite anime: ", bg='#FAEBD7', font=("Arial", 12))
    entry_label.grid(column=0, row=0)
    global anime
    anime= tk.StringVar()
    anime.set("")
    entry = tk.Entry(lower_frame, width=50, textvariable= anime, font=("Arial", 12))
    entry.focus_set()
    entry.grid(column=3, row=0)
    submit_button = tk.Button(lower_frame, text="Submit", font=("Arial", 12), command = check) 
    submit_button.grid(column=2, row=4)
    #result = get_recommendation("Kimetsu no Yaiba")
    #result2 = get_recommendation("Yakusoku no Neverland")
    
main()
root.mainloop()