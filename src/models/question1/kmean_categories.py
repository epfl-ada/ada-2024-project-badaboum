import spacy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator
from src.models.question1.datasets_loading import load_oscar_movies_all_categories


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


def get_optimal_k(sses: list, k_list: list):
    kneedle = KneeLocator(k_list, sses, curve="convex", direction="decreasing")
    return kneedle.elbow


def get_clusters(embedded_categories: list, k_list: list):
    sses = kmeans_sse(embedded_categories, k_list)
    optimal_k = get_optimal_k(sses, k_list)
    labels = perform_kmeans(embedded_categories, optimal_k)
    return labels, optimal_k


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


def find_best_k(k_start: int = 2, k_end: int = 30, min_samples: int = 10):
    embedded_categories, _ = get_embedded_categories(min_samples)

    # Get the optimal number of clusters
    k_list = range(k_start, k_end)
    sses = kmeans_sse(embedded_categories, k_list)
    optimal_k = get_optimal_k(sses, k_list)

    print(f"Optimal number of clusters: {optimal_k}")

    return optimal_k, sses


def print_clusters(k, min_samples: int = 10):
    embedded_categories, categories = get_embedded_categories(min_samples)
    labels = perform_kmeans(embedded_categories, k)

    for i in range(k):
        print(f"Cluster {i}:")
        for index in np.where(labels == i)[0]:
            print(categories[index], end=", ")
        print("\n")
