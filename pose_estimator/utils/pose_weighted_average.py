import numpy as np
import numpy.matlib as npm
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq
from transforms3d.euler import quat2euler
from transforms3d.quaternions import rotate_vector

__all__ = ["get_weighted_average", "get_kmeans_center"]


def averageQuaternions(Q):
    # Number of quaternions to average
    M = Q.shape[0]
    A = npm.zeros(shape=(4, 4))

    for i in range(0, M):
        q = Q[i, :]
        # multiply q with its transposed version q' and add A
        A = np.outer(q, q) + A

    # scale
    A = (1.0 / M) * A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:, eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:, 0].A1)


def weightedAverageQuaternions(Q, w):
    # Number of quaternions to average
    M = Q.shape[0]
    A = npm.zeros(shape=(4, 4))
    weightSum = 0

    for i in range(0, M):
        q = Q[i, :]
        A = w[i] * np.outer(q, q) + A
        weightSum += w[i]
    # scale
    A = (1.0 / weightSum) * A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:, eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:, 0].A1)


def get_weighted_average(poses):
    """given array of x,y,z,qw,qx,qy,qz, return the weighted average of them"""
    quats = poses[:, 3:]
    weights = np.arange(50, 50 + quats.shape[0], dtype=float)
    weights /= sum(weights)
    trans = poses[:, :3].T @ weights
    quat = weightedAverageQuaternions(quats, weights)
    return [*trans, *quat]


def get_average(poses):
    """given array of x,y,z,qw,qx,qy,qz, return the weighted average of them"""
    quats = poses[:, 3:]
    trans = np.mean(poses[:, :3], axis=0)
    quat = averageQuaternions(quats)
    return [*trans, *quat]


def get_kmeans_center(poses, n_clusters=2):
    if len(poses) == 1:
        return poses[0]
    angles = np.array([quat2euler(q) for q in poses[:, 3:]])
    X = np.concatenate([poses[:, :3], angles], axis=1)
    centroids, _ = kmeans(X, n_clusters)
    idx, _ = vq(X, centroids)
    best_cluster = np.argmax(np.bincount(idx))
    # second_best = np.argsort(np.max(x, axis=0))[-2]
    return get_weighted_average(poses[idx == best_cluster])


if __name__ == "__main__":
    import timeit

    import matplotlib.pyplot as plt
    import pandas as pd
    from transforms3d.euler import quat2euler

    np.set_printoptions(precision=3, suppress=True)
    # poses_full = pd.read_csv("/home/saber/xingyu/bbauv_ws/debug_poses.csv", header=None).iloc[:,1:].to_numpy()
    poses_full = (
        pd.read_csv(
            "/home/nvidia/catkin_ws/src/image_matching/debug_poses.csv",
            header=None,
        )
        .iloc[:, 1:]
        .to_numpy()
    )
    # poses_full = pd.read_csv("/home/nvidia/catkin_ws/src/image_matching/debug_poses_2.csv", header=None).iloc[:,1:].to_numpy()
    # poses_full = pd.read_csv("/home/nvidia/catkin_ws/src/image_matching/debug_poses_thresh_3.csv", header=None).iloc[:,1:].to_numpy()
    # poses_full = pd.read_csv("/home/nvidia/catkin_ws/src/image_matching/debug_poses_5_nonrect.csv", header=None).iloc[:,1:].to_numpy()
    for i in range(0, 1):
        poses = poses_full[int(i / 10 * len(poses_full)) :]
        # mean = get_weighted_average(poses)
        # mean = get_average(poses)
        mean = get_kmeans_center(poses)
        mean_angle = quat2euler(mean[3:])

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        for vector in poses:
            v = rotate_vector((0, 0, 1), vector[3:])
            vlength = 0.1
            ax.quiver(
                vector[0],
                vector[1],
                vector[2],
                v[0],
                v[1],
                v[2],
                pivot="tail",
                length=vlength,
                color="b",
            )

        mean_v = rotate_vector((0, 0, 1), mean[3:])
        ax.quiver(
            mean[0],
            mean[1],
            mean[2],
            mean_v[0],
            mean_v[1],
            mean_v[2],
            length=0.2,
            color="r",
            pivot="tail",
            arrow_length_ratio=0.2,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.gca().set_aspect("auto", adjustable="box")

        plt.show()
        print(np.array(mean)[:3], np.array(mean_angle) * 180 / np.pi)
    print(
        timeit.timeit(
            "get_weighted_average(poses)", number=100, globals=globals()
        )
    )
    print(timeit.timeit("get_average(poses)", number=100, globals=globals()))
    print(
        timeit.timeit(
            "get_kmeans_center(poses)", number=100, globals=globals()
        )
    )
