import tensorcircuit as tc
import optax
import jax.numpy as jnp
from scipy.cluster.hierarchy import linkage, fcluster
import jax
from jax import lax
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import time
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

plt.rcParams["font.family"] = "serif"
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rc('font', size=14)

K = tc.set_backend('jax')
key = jax.random.PRNGKey(42)
tf.random.set_seed(42)

n_world = 10
local_rounds = 1
dataset = 'mnist'
# dataset = 'fashion'
readout_mode = 'softmax'
# readout_mode = 'sample'
# 原始编码、均值编码、半值编码、振幅编码和角度编码
encoding_mode = 'vanilla'
# encoding_mode = 'mean'
# encoding_mode = 'half'
# encoding_mode = 'amplitude'
# encoding_mode = 'angle'


n = 8
n_node = 8
k = 48


def filter(x, y, class_list):
    keep = jnp.zeros(len(y)).astype(bool)
    for c in class_list:
        keep = keep | (y == c)
    x, y = x[keep], y[keep]
    y = jax.nn.one_hot(y, n_node)
    return x, y


# 量子分类器
def clf(params, c, k):
    for j in range(k):
        for i in range(n - 1):
            c.cnot(i, i + 1)
        for i in range(n):
            c.rx(i, theta=params[3 * j, i])
            c.rz(i, theta=params[3 * j + 1, i])
            c.rx(i, theta=params[3 * j + 2, i])
    return c


def clf1(params, c, k):
    for j in range(k):
        for i in range(n - 1):
            c.cnot(i, i + 1)
            c.swap(i, (i + 2) % n)
        for i in range(n):
            c.ry(i, theta=params[4 * j, i])
            c.rz(i, theta=params[4 * j + 1, i])
            c.rx(i, theta=params[4 * j + 2, i])
            c.rz(i, theta=params[4 * j + 3, i])
        for i in range(n - 1):
            c.cnot(i, i + 1)
    return c


def prune_and_quantize_params_with_median_threshold(params, quantization_levels=48):
    prune_threshold = jnp.mean(jnp.abs(params))
    pruned_params = jnp.where(jnp.abs(params) < prune_threshold, 0, params)
    non_zero_params = pruned_params[pruned_params != 0]

    if non_zero_params.size > 0:
        mean_val = jnp.mean(non_zero_params)
        std_val = jnp.std(non_zero_params)
        min_val = mean_val - 2 * std_val
        max_val = mean_val + 2 * std_val
        thresholds = jnp.linspace(min_val, max_val, quantization_levels + 1)
        quantized_params = jnp.zeros_like(pruned_params)
        for i in range(quantization_levels):
            lower_bound = thresholds[i]
            upper_bound = thresholds[i + 1]
            level_value = (lower_bound + upper_bound) / 2
            quantized_params = jnp.where((pruned_params >= lower_bound) & (pruned_params < upper_bound), level_value,
                                         quantized_params)
    else:
        quantized_params = pruned_params

    return quantized_params


# 标量量化
def scalar_quantize_params(params, quantization_levels=48):
    min_val = jnp.min(params)
    max_val = jnp.max(params)
    scale = (max_val - min_val) / (quantization_levels - 1)
    quantized_params = jnp.round((params - min_val) / scale) * scale + min_val
    return quantized_params


# 概率量化
def probabilistic_quantize_params(params, quantization_levels=128):
    # 确定参数的最小值和最大值
    min_val, max_val = jnp.min(params), jnp.max(params)
    # 计算分布区间
    levels = jnp.linspace(min_val, max_val, quantization_levels)
    # 初始化量化后的参数
    quantized_params = jnp.zeros_like(params)
    # 对每个参数进行量化
    for i in range(1, len(levels)):
        lower = levels[i - 1]
        upper = levels[i]
        # 计算当前区间的中心值作为量化值
        center_value = (lower + upper) / 2
        # 概率量化逻辑，这里简化为直接量化到中心值
        quantized_params = jnp.where((params >= lower) & (params < upper), center_value, quantized_params)
    return quantized_params


def log_quantize_params(params, quantization_levels=128):
    min_val, max_val = jnp.min(jnp.abs(params)), jnp.max(jnp.abs(params))
    log_min, log_max = jnp.log(min_val), jnp.log(max_val)
    log_levels = jnp.linspace(log_min, log_max, quantization_levels)
    sign = jnp.sign(params)
    log_params = jnp.log(jnp.abs(params))
    indices = jnp.digitize(log_params, log_levels) - 1
    indices = jnp.clip(indices, 0, quantization_levels - 1)
    quantized_params = jnp.exp(log_levels[indices]) * sign
    return quantized_params


def uniform_adaptive_quantize(params, quantization_levels=256):
    min_val = jnp.min(params)
    max_val = jnp.max(params)
    range_val = max_val - min_val
    step = range_val / (quantization_levels - 1)
    quantized_params = jnp.round((params - min_val) / step) * step + min_val
    return quantized_params


# K-means 聚类量化 将输入参数的值量化为指定数量 通过量化，模型的大小和计算量都可以大幅减少
def kmeans_clustering_quantize(params, quantization_levels=512):
    params_flat = params.flatten().reshape((-1, 1))
    kmeans = KMeans(n_clusters=quantization_levels, random_state=0).fit(params_flat)
    labels = kmeans.predict(params_flat)
    quantized_params = kmeans.cluster_centers_[labels].reshape(params.shape)
    return quantized_params


def kmeans_clustering_quantize1(params, quantization_levels=256):
    params_flat = params.flatten().reshape((-1, 1))
    kmeans = MiniBatchKMeans(n_clusters=quantization_levels, random_state=0, batch_size=4096).fit(params_flat)
    labels = kmeans.predict(params_flat)
    quantized_params = kmeans.cluster_centers_[labels].reshape(params.shape)
    return quantized_params


def top_k_sparse(params, k=1150):
    # 扁平化参数数组并获取其绝对值
    flat_params = params.flatten()
    abs_flat_params = jnp.abs(flat_params)

    # 使用jax.lax.top_k直接找到最大的k个值及其索引
    top_k_values, top_k_indices = lax.top_k(abs_flat_params, k)

    # 创建一个与扁平化参数同形状的全零数组
    sparse_flat_params = jnp.zeros_like(flat_params)

    # 在top_k_indices指定的位置上设置对应的原始参数值
    sparse_flat_params = sparse_flat_params.at[top_k_indices].set(flat_params[top_k_indices])

    # 将扁平化的稀疏参数数组重新塑形为原始参数数组的形状
    sparse_params = sparse_flat_params.reshape(params.shape)

    return sparse_params


# STC稀疏化函数
def stc_sparse(params, epsilon):
    mask = jnp.abs(params) > epsilon
    sparse_params = jnp.sign(params) * mask
    return sparse_params


def readout(c):
    if readout_mode == 'softmax':
        logits = []
        for i in range(n_node):
            logits.append(jnp.real(c.expectation([tc.gates.z(), [i, ]])))
        logits = jnp.stack(logits, axis=-1) * 10
        probs = jax.nn.softmax(logits)
    elif readout_mode == 'sample':
        wf = jnp.abs(c.wavefunction()[:n_node]) ** 2
        probs = wf / jnp.sum(wf)
    return probs


def loss(params, x, y, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return -jnp.mean(jnp.sum(y * jnp.log(probs + 1e-7), axis=-1))


loss = K.jit(loss, static_argnums=[3])


def accuracy(params, x, y, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return jnp.argmax(probs, axis=-1) == jnp.argmax(y, axis=-1)  # 使用经典网络的输出计算精度


accuracy = K.jit(accuracy, static_argnums=[3])

compute_loss = K.jit(K.vectorized_value_and_grad(loss, vectorized_argnums=[1, 2]), static_argnums=[3])
compute_accuracy = K.jit(K.vmap(accuracy, vectorized_argnums=[1, 2]), static_argnums=[3])


def pred(params, x, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return probs


pred = K.vmap(pred, vectorized_argnums=[1])


def optimized_hierarchical_clustering_with_cosine(params_list):
    # 转换参数列表为JAX数组
    params_array = jnp.array(params_list)

    # 将三维数组展平为二维数组
    flattened_params_array = params_array.reshape(params_array.shape[0], -1)

    # 计算参数列表的平均值作为基准v
    baseline = jnp.mean(flattened_params_array, axis=0)

    # 使用层次聚类算法将参数进行聚类
    Z = linkage(flattened_params_array, 'ward')

    # 使用fcluster函数根据距离阈值将聚类结果转换为簇标签
    labels = fcluster(Z, t=0.8, criterion='distance')

    # 找到与baseline最接近的簇中心
    cluster_centers = [jnp.mean(flattened_params_array[labels == i], axis=0) for i in range(1, jnp.max(labels) + 1)]
    closest_cluster_index = jnp.argmin(jnp.array([cosine_distance(center, baseline) for center in cluster_centers]))

    # 找到相似簇的索引
    similar_indices = jnp.where(labels == closest_cluster_index + 1)[0]

    # 计算相似簇的平均值
    avg_params = jnp.mean(params_array[similar_indices], axis=0)

    return avg_params


# 定义余弦距离的函数
def cosine_distance(u, v):
    # 计算两个向量的余弦相似度
    cosine_similarity = jnp.dot(u, v) / (jnp.linalg.norm(u) * jnp.linalg.norm(v))
    # 计算余弦距离
    cosine_distance = 1 - cosine_similarity
    return cosine_distance


if __name__ == '__main__':
    # numpy data
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'fashion':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    ind = y_test == 9
    x_test, y_test = x_test[~ind], y_test[~ind]
    ind = y_test == 8
    x_test, y_test = x_test[~ind], y_test[~ind]
    ind = y_train == 9
    x_train, y_train = x_train[~ind], y_train[~ind]
    ind = y_train == 8
    x_train, y_train = x_train[~ind], y_train[~ind]

    x_train = x_train / 255.0
    # 编码成的量子比特数和客户端的数量保持一致
    if encoding_mode == 'vanilla':
        mean = 0
    elif encoding_mode == 'mean':
        mean = jnp.mean(x_train, axis=0)
    elif encoding_mode == 'half':
        mean = 0.5
    elif encoding_mode == 'angle':
        mean = 0
        # 使用角度编码
        x_train = (x_train * 2 * jnp.pi) % (2 * jnp.pi)
        x_test = (x_test * 2 * jnp.pi) % (2 * jnp.pi)
    elif encoding_mode == 'amplitude':
        mean = 0
        # 使用振幅编码
        x_train = jnp.sqrt(x_train)
        x_test = jnp.sqrt(x_test)
    x_train = x_train - mean
    x_train = tf.image.resize(x_train[..., tf.newaxis], (int(2 ** (n / 2)), int(2 ** (n / 2)))).numpy()[..., 0].reshape(
        -1, 2 ** n)
    x_train = x_train / jnp.sqrt(jnp.sum(x_train ** 2, axis=-1, keepdims=True))

    x_test = x_test / 255.0
    x_test = x_test - mean
    x_test = tf.image.resize(x_test[..., tf.newaxis], (int(2 ** (n / 2)), int(2 ** (n / 2)))).numpy()[..., 0].reshape(
        -1, 2 ** n)
    x_test = x_test / jnp.sqrt(jnp.sum(x_test ** 2, axis=-1, keepdims=True))
    y_test = jax.nn.one_hot(y_test, n_node)

    class_test_loss = []
    class_test_acc = []
    class_train_acc = []
    class_train_loss = []
    time_list = []

    for n_class in jnp.arange(2, 3):
        world_train_loss = []
        world_train_acc = []
        world_test_loss = []
        world_test_acc = []
        for world in tqdm(range(n_world)):

            params_list = []
            opt_state_list = []
            data_list = []
            iter_list = []
            # 记录开始时间
            start_time = time.time()
            for node in range(n_node - 1):
                # 保留n_class个标签
                # 用于随着non-iid的程度降低，查看精度的变化
                # n_class_nodes = [6, 6, 1, 2, 3, 5, 4]
                # n_class_node = n_class_nodes[node]  # 新增：使用保存的每个节点的类别数

                x_train_node, y_train_node = filter(x_train, y_train,
                                                    [(node + i) % n_node for i in range(n_class)])
                # x_train_node, y_train_node = x_train, jax.nn.one_hot(y_train, n_node)
                data = tf.data.Dataset.from_tensor_slices((x_train_node, y_train_node)).batch(128)
                data_list.append(data)
                iter_list.append(iter(data))

                key, subkey = jax.random.split(key)
                params = jax.random.normal(subkey, (3 * k, n))
                # 初始化两个优化器
                opt = optax.adam(learning_rate=1e-2)
                opt_state = opt.init(params)
                params_list.append(params)
                opt_state_list.append(opt_state)

            loss_list = []
            acc_list = []

            for e in tqdm(range(5), leave=False):
                # 全局训练周期中的批次数
                for b in range(50):
                    weights = []
                    performance_metrics = []
                    for node in range(n_node - 1):
                        try:
                            x, y = next(iter_list[node])
                        except StopIteration:
                            iter_list[node] = iter(data_list[node])
                            x, y = next(iter_list[node])
                        x = x.numpy()
                        y = y.numpy()
                        # 增加本地训练的轮数，每个节点都有更多的机会来适应其本地数据分布，从而有助于减轻非独立同分布（Non-IID）数据的影响。
                        for _ in range(local_rounds):
                            loss_val, grad_val = compute_loss(params_list[node], x, y, k)

                            updates, opt_state_list[node] = opt.update(grad_val, opt_state_list[node],
                                                                       params_list[node])
                            params_list[node] = optax.apply_updates(params_list[node], updates)

                        # 收集所有客户端的模型参数
                        weights.append(params_list[node])
                        # 计算每个客户端模型的性能指标，例如准确率
                        performance_metric = compute_accuracy(params_list[node], x_test[:1024], y_test[:1024], k).mean()
                        performance_metrics.append(performance_metric)

                    # 先训练5轮，再进行层次聚类
                    quantized_params_list = []
                    for weight in weights:
                        # 使用概率量化处理参数
                        # quantized_params = probabilistic_quantize_params(weight)
                        # quantized_params = prune_and_quantize_params_with_median_threshold(weight)
                        # quantized_params = scalar_quantize_params(weight)
                        # quantized_params = recursive_bisection_quantize_params(weight)
                        # quantized_params = kmeans_clustering_quantize(weight)
                        quantized_params = kmeans_clustering_quantize(weight)
                        # top_k_sparse(weight)
                        quantized_params_list.append(quantized_params)
                    # 再使用层次聚类
                    avg_params = optimized_hierarchical_clustering_with_cosine(quantized_params_list)
                    # avg_params = jnp.mean(jnp.stack(params_list, axis=0), axis=0)
                    for node in range(n_node - 1):
                        params_list[node] = avg_params

                    if b % 10 == 0:
                        avg_loss = jnp.mean(compute_loss(avg_params, x_test[:1024], y_test[:1024], k)[0])
                        loss_list.append(avg_loss)
                        acc_list.append(compute_accuracy(avg_params, x_test[:1024], y_test[:1024], k).mean())
                        tqdm.write(
                            f"world {world}, epoch {e}, batch {b}/{50}: loss {avg_loss}, accuracy {acc_list[-1]}")
            # 记录结束时间
            end_time = time.time()
            # 计算并打印通信时间
            communication_time = end_time - start_time
            print(f"Communication time: {communication_time} seconds")
            time_list.append(communication_time)

            test_acc = jnp.mean(pred(avg_params, x_test[:1024], k).argmax(axis=-1) == y_test[:1024].argmax(axis=-1))
            test_loss = -jnp.mean(jnp.log(pred(avg_params, x_test[:1024], k)) * y_test[:1024])

            world_test_loss.append(test_loss)
            world_test_acc.append(test_acc)
            world_train_loss.append(loss_list)
            world_train_acc.append(acc_list)
            tqdm.write(f"world {world}: test loss {test_loss}, test accuracy {test_acc}")
        avg_test_loss = jnp.mean(jnp.array(world_test_loss), axis=0)
        avg_test_acc = jnp.mean(jnp.array(world_test_acc), axis=0)
        std_test_loss = jnp.std(jnp.array(world_test_loss), axis=0)
        std_test_acc = jnp.std(jnp.array(world_test_acc), axis=0)
        # 输出每轮通信的时间戳，计算通信加速比
        print(time_list)
        print(
            f'n_class {n_class}, test loss: {avg_test_loss}+-{std_test_loss}, test acc: {avg_test_acc}+-{std_test_acc}')
        class_test_loss.append(world_test_loss)
        class_test_acc.append(world_test_acc)
        class_train_acc.append(world_train_acc)
        class_train_loss.append(world_train_loss)

    os.makedirs(f'./{dataset}/Q-FedAvg-noniid-2-yu3', exist_ok=True)
    jnp.save(f'./{dataset}/Q-FedAvg-noniid-2-yu3/test_loss.npy', class_test_loss)
    jnp.save(f'./{dataset}/Q-FedAvg-noniid-2-yu3/test_acc.npy', class_test_acc)
    jnp.save(f'./{dataset}/Q-FedAvg-noniid-2-yu3/train_acc.npy', class_train_acc)
    jnp.save(f'./{dataset}/Q-FedAvg-noniid-2-yu3/train_loss.npy', class_train_loss)
