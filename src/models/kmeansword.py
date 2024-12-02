import pandas as pd
import spacy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

#clusters are less visible with seaborn
#sns.set_theme()


nlp = spacy.load("en_core_web_md")


df = pd.read_csv('./data/oscar_movies_all_categories.csv')
categories = df['oscar_category'].unique()
categories = [cat.split() for cat in categories]

#spacy use an embedding vector size of 300
vectors = np.zeros((len(categories), 300))
representative_labels = []




for i, cat in enumerate(categories):

    for subcat in cat:
        vectors[i, :] += nlp(subcat.strip()).vector

    #Mean vector
    vectors[i] /= len(cat)

    #Normalize vectors for k-means performance
    vectors[i] /= np.linalg.norm(vectors[i])

    representative_labels.append(" ".join(cat))


# Reduce dimensions for visualization
tsne = TSNE(n_components=2, random_state=0)
reduced_dim = tsne.fit_transform(vectors)


def plot_n_cluster(X, n_clusters):
    """
    Plot the clustering results of different number of clusters

    X: 2d arrays (embedding vectors)
    n_clusters: list of int, the number of clusters we compute for
    """

    for k in n_clusters:
        kmean = KMeans(n_clusters=k, random_state=42).fit(X)


        plt.figure(figsize=(10, 8))
        plt.grid(linestyle='--')
        plt.scatter(X[:, 0], X[:, 1], c=kmean.labels_, alpha=0.9)
        for c in kmean.cluster_centers_:
                plt.scatter(c[0], c[1], marker="+", color="red")


        for i, label in enumerate(representative_labels):
            #dont saturate the image:
            #alternative:
            #if i % 5 == 0:
            if label.count(" ") <= 1:
                plt.annotate(label, (X[i, 0], X[i, 1]), fontsize=8, alpha=0.8)

        plt.title('IMDB Oscar categories clustering')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.show()



def plot_silhouette(X, n_clusters):
    """
    Inspired from lab 8
    Plot the clustering results of different number of clusters

    X: 2d arrays (embedding vectors)
    n_clusters: list of int, the number of clusters we compute for
    """



    silhouettes = [0]*len(n_clusters)

    # Try multiple k
    for i,k in enumerate(n_clusters):
        
        labels = KMeans(n_clusters=k, random_state=10).fit_predict(X)
        
        score = silhouette_score(X, labels)
        silhouettes[i] = score
        


    
    plt.plot(n_clusters, silhouettes)
    plt.xlabel("K")
    plt.ylabel("Silhouette score")
    plt.grid(linestyle='--')
    plt.show()




     



def plot_sse(X, n_clusters):
    """
    Inspired from lab 8
    Plot the sse of different number of clusters

    X: 2d arrays (embedding vectors)
    n_clusters: list of int, the number of clusters we compute for
    """

    sse = [0]*len(n_clusters)
    for i,k in enumerate(n_clusters):
        # Assign the labels to the clusters
        kmeans = KMeans(n_clusters=k, random_state=10).fit(X)
        sse[i] = kmeans.inertia_

    sse = pd.DataFrame(sse)
    # Plot the data
    plt.plot(n_clusters, sse)
    plt.xlabel("K")
    plt.ylabel("Sum of Squared Errors")
    plt.grid(linestyle='--')
    plt.show()
    
    

n_clusters = range(3, 16)

#suggests k = 7
#plot_silhouette(reduced_dim, n_clusters)
#plot_sse(reduced_dim, n_clusters)
#plot_n_cluster(reduced_dim, [7])





### DBSCAN ###



#after seeing the results, dbscan might be more suited:

def dbscan_clustering(X, eps, min_samples):
    """
    Perform DBSCAN clustering and plot the results

    X: 2d arrays (embedding vectors)
    eps: float, maximum distance between two samples to be considered as neighbors
    min_samples: int, minimum samples required to form a dense region
    """


    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    #cluster assignments
    labels = db.labels_

    plt.figure(figsize=(10, 8))
    plt.grid(linestyle='--')
    unique_labels = set(labels)
    print(len(unique_labels))

    for label in unique_labels:
        #label == -1 means no cluster was found, I tried to find eps, min_samples s.t it does not happen
        color = 'black' if label == -1 else plt.cm.jet(float(label) / max(unique_labels))
        plt.scatter(X[labels == label, 0], X[labels == label, 1], c=[color], alpha=0.7)


    for i, text in enumerate(representative_labels):
        #dont saturate the image:
        #alternative:
        #if i % 5 == 0:
        if text.count(" ") <=1:
            plt.annotate(text, (X[i, 0], X[i, 1]), fontsize=8, alpha=0.8)

    plt.title('IMDB Oscar categories clustering')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.show()

    

#to tune to match the existing categories
eps = 1.3
min_samples = 2

dbscan_clustering(reduced_dim, eps, min_samples)