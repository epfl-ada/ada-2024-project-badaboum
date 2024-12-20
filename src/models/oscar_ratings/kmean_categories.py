import spacy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from src.models.oscar_ratings.datasets_loading import load_oscar_movies_all_categories


def embed_categories(categories: list):
    nlp = spacy.load("en_core_web_md")
    vectors = np.zeros((len(categories), 300))

    for i, cat in enumerate(categories):
        splitted_cat = cat.split()

        for subcat in splitted_cat:
            vectors[i, :] += nlp(subcat.strip()).vector

        # # Mean vector
        vectors[i] /= len(splitted_cat)

        # Normalize vectors for k-means performance
        vectors[i] /= np.linalg.norm(vectors[i])

    return vectors


def perform_kmeans(embedded_categories: list, k: int, random_state: int = 42):
    kmean = KMeans(n_clusters=k, random_state=random_state).fit(embedded_categories)
    return kmean.labels_


def kmeans_sse(embedded_categories: list, k_list: list, random_state: int = 42):
    sses = []
    for k in k_list:
        sse = (
            KMeans(n_clusters=k, random_state=random_state)
            .fit(embedded_categories)
            .inertia_
        )
        sses.append(sse)

    return sses


def reduce_dim(embedded_categories: list, out_dim: int):
    # Reduce dim with PCA
    pca = PCA(n_components=out_dim)
    return pca.fit_transform(embedded_categories)


def get_embedded_categories(min_samples: int = 10):
    oscar_movies = load_oscar_movies_all_categories()

    # Get the main categories
    categories = oscar_movies["oscar_category"].unique()
    categories = [
        cat
        for cat in categories
        if (oscar_movies["oscar_category"] == cat).sum() > min_samples
    ]

    return embed_categories(categories), categories


def print_clusters(k, min_samples: int = 10):
    embedded_categories, categories = get_embedded_categories(min_samples)
    labels = perform_kmeans(embedded_categories, k)

    for i in range(k):
        print(f"Cluster {i}:")
        for index in np.where(labels == i)[0]:
            print(categories[index], end=", ")
        print("\n")
