import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def matr(data):
    data = pd.DataFrame(data)
    corr_matr = data.corr()
    sns.set_style('darkgrid')
    sns.heatmap(corr_matr, cmap='coolwarm', annot=True)
    plt.title('Корреляционная матрица')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()


def main():
    data = pd.read_csv('data.csv')
    print(data.describe())
    matr(data)

    data_2d = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(data))

    optimal = 3

    km = KMeans(n_clusters=optimal, n_init=3, max_iter=100, random_state=42)
    km.fit(data_2d)
    labels = km.labels_

    plt.figure(figsize=(10, 8))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], s=1, c=labels, cmap="Set3", alpha=0.5)
    plt.title(f'Кластеризация данных (K={optimal})')
    plt.tight_layout()
    plt.savefig('clusters_result.png', dpi=100, bbox_inches='tight')
    plt.close()

    print(f"Кластеризация завершена. K={optimal}")


if __name__ == '__main__':
    main()