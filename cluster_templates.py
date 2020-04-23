"""
This python script pre-possess the templates label data and apply KMeans clustering.

@:author: Mitul Prajapati (mituldprajapati@gmail.com)
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from textblob import Word
import re
import os


def visualize_best_k_kmeans(x_train, cluster_range=(2, 10), save_plot=False):
    """
    Visualize best number using elbow method
    :param x_train: train data for Kmeans
    :param cluster_range: range of k to be assessed
    :param save_plot: if true then save plot to file named 'best_k_clusters.png'
    :return: None
    """

    cluster_inertia_list = []
    silhouette_score_list = []

    cluster_range = range(cluster_range[0], cluster_range[1])
    for i in cluster_range:
        # K means clustering using the term vector

        kmeans = KMeans(n_clusters=i, random_state=90).fit(x_train)
        clusters = kmeans.predict(x_train)
        cluster_inertia_list.append(kmeans.inertia_)
        silhouette_score_list.append(silhouette_score(x_train, clusters))

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(cluster_range, cluster_inertia_list, 'b-', label='inertia')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(cluster_range, silhouette_score_list, 'r-', label='silhouette')
    ax[1].legend()
    ax[1].grid(True)
    fig.suptitle('Find best k (Elbow Method)')
    plt.legend()

    if save_plot:
        plt.savefig('best_k_clusters.png', bbox_inches='tight', dpi=300)
    else:
        plt.show()


def apply_svd(data, n):
    """
    This method transforms the data and lower the number of components using SVD.
    :param data: feature_vector
    :param n: components in output feature_vector, if feature_vector is of shape i X j, it will converted to i X n
    :return: transformed feature vector
    """
    svd = TruncatedSVD(n_components=n, random_state=40)
    transformed_data = svd.fit_transform(data)
    return transformed_data


def main():
    # config
    pd.set_option('display.max_columns', 25)
    data_dir = 'data'

    # load data
    data = pd.read_csv(os.path.join(data_dir, 'templates_data.csv'), sep='|')

    print('[INFO]: Data loaded successfully from {0}'.format(os.path.join(data_dir, 'templates_data.csv')))

    # apply lemmatizer, lower and remove numbers from keywords
    data['keywords'] = data['keywords'].apply(lambda x: ' '.join(re.sub('[0-9]', '', Word(i.lower()).lemmatize())
                                                                 .strip() for i in eval(x)))

    print('[INFO]: Data pre-processing done')

    # apply TF-IDF vectorizer
    tfidf = TfidfVectorizer(analyzer='word')
    x_train_tfidf = tfidf.fit_transform(data['keywords'])

    # remove blank rows which doesn't have any text labels
    non_zero_indices = [i for i, x in enumerate(x_train_tfidf.toarray().any(1)) if x]
    x_train = x_train_tfidf[non_zero_indices]
    data = data.iloc[non_zero_indices]

    # transform data using Single Value Decomposition
    x_train = apply_svd(x_train, 100)

    # get ideal k
    min_num_cluster = 2  # set lower limit for number of clusters
    max_num_cluster = 12  # set higher limit for number of clusters
    print('[INFO]: Training Kmeans ...')
    visualize_best_k_kmeans(x_train, (min_num_cluster, max_num_cluster), save_plot=False)

    # get clusters using Kmeans
    kmeans = KMeans(n_clusters=9, random_state=90).fit(x_train)

    print('[INFO]: Training Done')

    # store clustered data
    data['cluster'] = kmeans.predict(x_train)
    data.to_csv(os.path.join(data_dir, 'clustered_templates_data.csv'), sep='|')
    print('[INFO]: Cluster data successfully saved at {0}'
          .format(os.path.join(data_dir, 'clustered_templates_data.csv')))


if __name__ == '__main__':
    main()
