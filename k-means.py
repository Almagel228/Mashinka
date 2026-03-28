import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def find_k(data, n=10):
    sc = []

    for i in range(2, n + 1):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(data)
        labels = km.labels_
        score = silhouette_score(data, labels)
        sc.append(score)
        print(f"K={i}: силуэт = {score:.3f}")

    max_value = max(sc)
    optimal_k = sc.index(max_value) + 2
    return optimal_k


def my_kmeans(X, k, num_iters, eps_peps):
    np.random.seed(42)

    colors = ['red', 'blue', 'green', 'purple', 'orange',
              'brown', 'pink', 'gray', 'olive', 'cyan']

    centroind_ids = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[centroind_ids].copy()

    step = 0

    for i in range(num_iters):
        step += 1

        clusters = {i: [] for i in range(k)}

        for x in X:
            dist = np.linalg.norm(centroids - x, axis=1)
            clusters_ind = dist.argmin()
            clusters[clusters_ind].append(x)

        new_centroids_dict = {}
        for cluster_id in clusters:
            if len(clusters[cluster_id]) > 0:
                new_centroids_dict[cluster_id] = np.mean(clusters[cluster_id], axis=0)
            else:
                new_centroids_dict[cluster_id] = centroids[cluster_id]

        new_centroids_dict = dict(sorted(new_centroids_dict.items()))
        new_centroids = np.array(list(new_centroids_dict.values()))

        plt.figure(figsize=(10, 8))

        for cluster_id in clusters:
            if len(clusters[cluster_id]) > 0:
                points = np.array(clusters[cluster_id])
                plt.scatter(points[:, 0], points[:, 1],
                            s=50, c=colors[cluster_id],
                            alpha=0.6, label=f'Кластер {cluster_id + 1}')

        plt.scatter(centroids[:, 0], centroids[:, 1],
                    s=200, marker='X', c='black',
                    label='Старые центроиды', alpha=0.7)

        plt.scatter(new_centroids[:, 0], new_centroids[:, 1],
                    s=300, marker='*', c='red',
                    label='Новые центроиды', edgecolors='black')

        for j in range(k):
            plt.arrow(centroids[j, 0], centroids[j, 1],
                      new_centroids[j, 0] - centroids[j, 0],
                      new_centroids[j, 1] - centroids[j, 1],
                      head_width=0.05, head_length=0.1,
                      fc='black', ec='black', alpha=0.7)

        plt.title(f'K-means: Шаг {step}', fontsize=14)
        plt.xlabel('Первая главная компонента')
        plt.ylabel('Вторая главная компонента')
        plt.legend()
        plt.grid(True, alpha=0.3)

        filename = f'kmeans_step_{step:03d}.png'
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"Шаг {step} сохранен в {filename}")
        plt.close()

        dist_between_centroids = new_centroids - centroids
        shift = np.linalg.norm(dist_between_centroids)
        print(f"  Сдвиг центроидов: {shift:.4f}")

        if shift <= eps_peps:
            break

        centroids = new_centroids.copy()

    return centroids, clusters


def main():
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    print("1)Поиск оптимального K\n")
    k = find_k(X)
    print(f"\nОптимальное кол-во кластеров: {k}")

    X_2d = PCA(n_components=2).fit_transform(X)

    print(f"\n2)K-means с K={k}\n")
    centroids, clusters = my_kmeans(X_2d, k, 100, 0.0001)

    print("\nФинальные центроиды:")
    for i, c in enumerate(centroids):
        print(f"  Кластер {i + 1}: ({c[0]:.3f}, {c[1]:.3f})")


if __name__ == "__main__":
    main()