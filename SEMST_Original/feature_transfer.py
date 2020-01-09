import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans as KMeansCPU
from skimage import color


# matching = {
#     (2, 2): {0: [0], 1: [1]},
#     (2, 3): {0: [0], 1: [1, 2]},
#     (2, 4): {0: [0, 2], 1: [1, 3]},
#     (2, 5): {0: [0, 1, 3], 1: [2, 4]},
#
#     (3, 2): {0: [0], 1: [0, 1], 2: [1]},
#     (3, 3): {0: [0, 1], 1: [1, 2], 2: [2]},
#     (3, 4): {0: [0, 1], 1: [2, 3], 2: [3]},
#     (3, 5): {0: [0, 1], 1: [2, 3], 2: [3, 4]},
#
#     (4, 2): {0: [0], 1: [0], 2: [0, 1], 3: [1]},
#     (4, 3): {0: [0], 1: [1], 2: [1, 2], 3: [1, 2]},
#     (4, 4): {0: [0, 1], 1: [1], 2: [2, 3], 3: [3]},
#     (4, 5): {0: [0, 1], 1: [1], 2: [2, 4], 3: [3, 4]},
#
#     (5, 2): {0: [0], 1: [0], 2: [0], 3: [0, 1], 4: [0, 1]},
#     (5, 3): {0: [0], 1: [0, 2], 2: [2], 3: [1], 4: [1]},
#     (5, 4): {0: [0], 1: [1, 2], 2: [2], 3: [2, 3], 4: [2]},
#     (5, 5): {0: [0], 1: [1], 2: [1, 2], 3: [2, 4], 4: [3, 4]},
# }


matching = {
    (2, 2): {0: [0], 1: [1]},
    (2, 3): {0: [0], 1: [1, 2]},
    (2, 4): {0: [0, 2], 1: [1, 3]},
    (2, 5): {0: [0, 1, 3], 1: [2, 4]},

    (3, 2): {0: [0], 1: [0, 1], 2: [1]},
    (3, 3): {0: [0, 1], 1: [1, 2], 2: [2]},
    (3, 4): {0: [0, 1], 1: [2, 3], 2: [3]},
    (3, 5): {0: [0, 1], 1: [1, 2, 3], 2: [3, 4]},

    (4, 2): {0: [0], 1: [0], 2: [0, 1], 3: [0, 1]},
    (4, 3): {0: [0], 1: [0, 1], 2: [1, 2], 3: [0, 2]},
    (4, 4): {0: [0, 1], 1: [0, 1], 2: [2, 3], 3: [1, 3]},
    (4, 5): {0: [0, 1], 1: [1, 2], 2: [2, 3, 4], 3: [3, 4]},

    (5, 2): {0: [0], 1: [0], 2: [0], 3: [0, 1], 4: [0, 1]},
    (5, 3): {0: [0], 1: [0, 1], 2: [1, 2], 3: [1, 2], 4: [2]},
    (5, 4): {0: [0, 1], 1: [0, 2], 2: [1, 2], 3: [2, 3], 4: [1, 3]},
    (5, 5): {0: [0, 1], 1: [0, 2], 2: [1, 2], 3: [2, 4], 4: [3, 4]},
}


def cal_maxpool_size(w, h, count=3):
    if count == 3:
        w = np.ceil(np.ceil(np.ceil(w / 2) / 2) / 2)
        h = np.ceil(np.ceil(np.ceil(h / 2) / 2) / 2)
    elif count == 2:
        w = np.ceil(np.ceil(w / 2) / 2)
        h = np.ceil(np.ceil(h / 2) / 2)
    elif count == 1:
        w = np.ceil(w / 2)
        h = np.ceil(h / 2)
    else:
        raise ValueError
    return int(w), int(h)


class KMeansGPU:
    def __init__(self, n_clusters, device='cuda', tol=1e-4, init='kmeans++'):
        self.n_clusters = n_clusters
        self.device = device
        self.tol = tol
        self.init = init
        self._labels = None
        self._cluster_centers = None
        self.init = init

    def _initial_state(self, data):
        # initial cluster centers by kmeans++ or random
        if self.init == 'kmeans++':
            n, c = data.shape
            dis = torch.zeros((n, self.n_clusters), device=self.device)
            initial_state = torch.zeros((self.n_clusters, c), device=self.device)
            pr = np.repeat(1 / n, n)
            initial_state[0, :] = data[np.random.choice(np.arange(n), p=pr)]

            dis[:, 0] = torch.sum((data - initial_state[0, :]) ** 2, dim=1)

            for k in range(1, self.n_clusters):
                pr = torch.sum(dis, dim=1) / torch.sum(dis)
                initial_state[k, :] = data[np.random.choice(np.arange(n), 1, p=pr.to('cpu').numpy())]
                dis[:, k] = torch.sum((data - initial_state[k, :]) ** 2, dim=1)
        else:
            n = data.shape[0]
            indices = np.random.choice(n, self.n_clusters)
            initial_state = data[indices]

        return initial_state

    @staticmethod
    def pairwise_distance(data1, data2=None):
        # using broadcast mechanism to calculate pairwise ecludian distance of data
        # the input data is N*M matrix, where M is the dimension
        # we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
        # then a simple elementwise operation of A and B will handle
        # the pairwise operation of points represented by data
        if data2 is None:
            data2 = data1

        # N*1*M
        a = data1.unsqueeze(dim=1)
        # 1*N*M
        b = data2.unsqueeze(dim=0)

        dis = (a - b) ** 2.0
        # return N*N matrix for pairwise distance
        dis = dis.sum(dim=-1).squeeze()

        return dis

    def fit(self, data):
        data = data.to(torch.float32)
        cluster_centers = self._initial_state(data)

        while True:
            dis = self.pairwise_distance(data, cluster_centers)

            labels = torch.argmin(dis, dim=1)
            cluster_centers_pre = cluster_centers.clone()

            for index in range(self.n_clusters):
                selected = labels == index
                if selected.any():
                    selected = data[labels == index]
                    cluster_centers[index] = selected.mean(dim=0)
                else:
                    cluster_centers[index] = torch.zeros_like(cluster_centers[0], device=self.device)

            center_shift = torch.sum(torch.sqrt(torch.sum((cluster_centers - cluster_centers_pre) ** 2, dim=1)))

            if center_shift ** 2 < self.tol:
                break

        self._labels = labels
        self._cluster_centers = cluster_centers

    @property
    def labels_(self):
        return self._labels

    @property
    def cluster_centers_(self):
        return self._cluster_centers


def calc_k(image_path,
           device='cpu',
           max_cluster=5,
           threshold_min=0.1,
           threshold_max=0.7,
           show_img_and_cluster=False):
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    #     gb = 0.5 if max(w, h) < 1440 else 0
    w, h = cal_maxpool_size(w, h, 3)

    img = img.resize((w, h))
    #     img = img.filter(ImageFilter.GaussianBlur(gb))

    if show_img_and_cluster:
        plt.imshow(np.array(img))
        plt.show()

    img = color.rgb2lab(img).reshape(w * h, -1)

    k = 2
    if not isinstance(device, str) and device.type.startswith('cuda'):
        KMeans = KMeansGPU
        img = torch.from_numpy(img).to(device)

        k_means_estimator = KMeans(k, device=f'cuda:{device.index}')
        k_means_estimator.fit(img)
        labels = k_means_estimator.labels_
        previous_labels = k_means_estimator.labels_
        previous_cluster_centers = k_means_estimator.cluster_centers_

        while True:
            cnt = Counter(labels.to('cpu').tolist())
            if k <= max_cluster and (cnt.most_common()[-1][1] / (w * h) > threshold_min or cnt.most_common()[0][1] / (
                    w * h) > threshold_max):
                if cnt.most_common()[-2][1] / (w * h) < threshold_min:
                    labels = previous_labels
                    cluster_centers = previous_cluster_centers
                    k = k - 1
                    break
                k = k + 1
            else:
                if k > max_cluster:
                    labels = previous_labels
                    cluster_centers = previous_cluster_centers
                    k = k - 1
                else:
                    labels = k_means_estimator.labels_
                    cluster_centers = k_means_estimator.cluster_centers_
                break

            previous_labels = k_means_estimator.labels_
            previous_cluster_centers = k_means_estimator.cluster_centers_

            k_means_estimator = KMeans(k, device=f'cuda:{device.index}')
            k_means_estimator.fit(img)
            labels = k_means_estimator.labels_

        new_clusters = cluster_centers.norm(dim=1).argsort(descending=False).tolist()
        new_clusters = [new_clusters.index(j) for j in range(k)]

        new_labels = torch.zeros_like(labels)

        for i in range(k):
            new_labels[labels == i] = new_clusters[i]

        label = new_labels.reshape(h, w)

        if show_img_and_cluster:
            plt.imshow(label.clone().to('cpu'))
            plt.colorbar()
            plt.show()

    else:
        KMeans = KMeansCPU
        k_means_estimator = KMeans(k)
        k_means_estimator.fit(img)
        labels = k_means_estimator.labels_
        previous_labels = k_means_estimator.labels_
        previous_cluster_centers = k_means_estimator.cluster_centers_

        while True:
            cnt = Counter(labels)
            if k <= max_cluster and (cnt.most_common()[-1][1] / (w * h) > threshold_min or cnt.most_common()[0][1] / (
                    w * h) > threshold_max):
                if cnt.most_common()[-2][1] / (w * h) < threshold_min:
                    labels = previous_labels
                    cluster_centers = previous_cluster_centers
                    k = k - 1
                    break
                k = k + 1
            else:
                if k > max_cluster:
                    labels = previous_labels
                    cluster_centers = previous_cluster_centers
                    k = k - 1
                else:
                    labels = k_means_estimator.labels_
                    cluster_centers = k_means_estimator.cluster_centers_
                break

            previous_labels = k_means_estimator.labels_
            previous_cluster_centers = k_means_estimator.cluster_centers_

            k_means_estimator = KMeans(k)
            k_means_estimator.fit(img)
            labels = k_means_estimator.labels_

        labels = torch.tensor(labels)
        cluster_centers = torch.tensor(cluster_centers)

        new_clusters = cluster_centers.norm(dim=1).argsort(descending=False).tolist()
        new_clusters = [new_clusters.index(j) for j in range(k)]

        new_labels = torch.zeros_like(labels)

        for i in range(k):
            new_labels[labels == i] = new_clusters[i]

        label = new_labels.reshape(h, w)

        if show_img_and_cluster:
            plt.imshow(label)
            plt.colorbar()
            plt.show()
    return label


def labeled_whiten_and_color(f_c, f_s, alpha, clabel):
    try:
        cc, ch, cw = f_c.shape
        cf = (f_c * clabel).reshape(cc, -1)

        num_nonzero = torch.sum(clabel).item() / cc
        c_mean = torch.sum(cf, 1) / num_nonzero
        c_mean = c_mean.reshape(cc, 1, 1) * clabel

        cf = cf.reshape(cc, ch, cw) - c_mean
        cf = cf.reshape(cc, -1)

        c_cov = torch.mm(cf, cf.t()) / (num_nonzero - 1)
        c_u, c_e, c_v = torch.svd(c_cov)
        c_d = c_e.pow(-0.5)

        w_step1 = torch.mm(c_v, torch.diag(c_d))
        w_step2 = torch.mm(w_step1, (c_v.t()))
        whitened = torch.mm(w_step2, cf)

        sf = f_s
        sc, shw = sf.shape
        s_mean = torch.mean(f_s, 1, keepdim=True)
        sf = sf - s_mean

        s_cov = torch.mm(sf, sf.t()) / (shw - 1)
        s_u, s_e, s_v = torch.svd(s_cov)
        s_d = s_e.pow(0.5)

        c_step1 = torch.mm(s_v, torch.diag(s_d))
        c_step2 = torch.mm(c_step1, s_v.t())
        colored = torch.mm(c_step2, whitened).reshape(cc, ch, cw)

        colored = colored + s_mean.reshape(sc, 1, 1) * clabel
        colored_feature = alpha * colored + (1 - alpha) * (f_c * clabel)
    except:
        colored_feature = f_c * clabel

    return colored_feature
