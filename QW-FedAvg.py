import random

import tensorcircuit as tc
import optax
import jax.numpy as jnp
import jax
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import math

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
k = 36
local_rounds = 10


def filter(x, y, class_list):
    keep = jnp.zeros(len(y)).astype(bool)
    for c in class_list:
        keep = keep | (y == c)
    x, y = x[keep], y[keep]
    y = jax.nn.one_hot(y, n_node)
    return x, y


def clf(params, c, k):
    for j in range(k):
        for i in range(n - 1):
            c.cnot(i, i + 1)
        for i in range(n):
            c.rx(i, theta=params[3 * j, i])
            c.rz(i, theta=params[3 * j + 1, i])
            c.rx(i, theta=params[3 * j + 2, i])
    return c


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
    # reg = 1e-4  # 正则化系数
    # return -jnp.mean(jnp.sum(y * jnp.log(probs + 1e-7), axis=-1)) + reg * jnp.sum(params ** 2)
    return -jnp.mean(jnp.sum(y * jnp.log(probs + 1e-7), axis=-1))


loss = K.jit(loss, static_argnums=[3])


def accuracy(params, x, y, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return jnp.argmax(probs, axis=-1) == jnp.argmax(y, axis=-1)


accuracy = K.jit(accuracy, static_argnums=[3])

compute_loss = K.jit(K.vectorized_value_and_grad(loss, vectorized_argnums=[1, 2]), static_argnums=[3])
compute_accuracy = K.jit(K.vmap(accuracy, vectorized_argnums=[1, 2]), static_argnums=[3])


def pred(params, x, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return probs


pred = K.vmap(pred, vectorized_argnums=[1])

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

    n_class_nodes = []  # 新增：存储每个节点的类别数
    for node in range(n_node - 1):
        # 使用jax.random.randint生成一个随机数，表示这个客户端的数据类别数
        key, subkey = jax.random.split(key)
        n_class_node = jax.random.randint(subkey, shape=(), minval=1, maxval=7)  # 模拟真实世界的数据分布
        n_class_nodes.append(n_class_node)  # 新增：保存每个节点的类别数
    print(f'每个客户端的类别数：{n_class_nodes}')

    for n_class in jnp.arange(5, 6):
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
                n_class_node = n_class_nodes[node]  # 新增：使用保存的每个节点的类别数
                # 保留n_class_node个标签
                x_train_node, y_train_node = filter(x_train, y_train, [(node + i) % n_node for i in range(n_class_node)])
                # 保留n_class个标签
                # 用于随着non-iid的程度降低，查看精度的变化
                # x_train_node, y_train_node = filter(x_train, y_train, [(node + i) % n_node for i in range(n_class)])
                # x_train_node, y_train_node = x_train, jax.nn.one_hot(y_train, n_node)

                data = tf.data.Dataset.from_tensor_slices((x_train_node, y_train_node)).batch(128)
                data_list.append(data)
                iter_list.append(iter(data))

                key, subkey = jax.random.split(key)
                params = jax.random.normal(subkey, (3 * k, n))
                opt = optax.adam(learning_rate=1e-2)
                opt_state = opt.init(params)
                params_list.append(params)
                opt_state_list.append(opt_state)

            loss_list = []
            acc_list = []

            for e in tqdm(range(5), leave=False):
                # 全局训练周期中的批次数
                for b in range(50):
                    weights = []  # 新增：存储每个节点的权重
                    # 抽取全局共享数据
                    global_data_indices = jnp.array(random.sample(range(len(x_train)), 256))  # 将索引转换为JAX数组
                    global_x_data, global_y_data = x_train[global_data_indices], y_train[global_data_indices]
                    for node in range(n_node - 1):
                        try:
                            x, y = next(iter_list[node])
                        except StopIteration:
                            iter_list[node] = iter(data_list[node])
                            x, y = next(iter_list[node])
                        x = x.numpy()
                        y = y.numpy()
                        # 新增：将全局共享数据添加到每个节点的本地数据中
                        x = jnp.concatenate((x, global_x_data), axis=0)
                        y = jnp.concatenate((y, jax.nn.one_hot(global_y_data, n_node)), axis=0)
                        loss_vals = []  # 新增：存储每轮本地训练的损失
                        acc_vals = []  # 新增：存储每轮本地训练的准确率
                        # 增加本地训练的轮数，每个节点都有更多的机会来适应其本地数据分布，从而有助于减轻非独立同分布（Non-IID）数据的影响。
                        for _ in range(local_rounds):
                            loss_val, grad_val = compute_loss(params_list[node], x, y, k)
                            acc = compute_accuracy(params_list[node], x, y, k)  # 训练精度

                            updates, opt_state_list[node] = opt.update(grad_val, opt_state_list[node],
                                                                       params_list[node])
                            params_list[node] = optax.apply_updates(params_list[node], updates)
                            loss_vals.append(loss_val)  # 新增：记录每轮本地训练的损失
                            # acc_vals.append(acc)  # 记录每轮本地训练的精度
                        weights.append(jnp.mean(jnp.array(loss_vals)))  # 新增： 只考 虑损失的均值
                        # weights.append(jnp.mean(jnp.array(acc_vals)))

                    weights = jnp.array(weights)  # 新增：将weights转换为DeviceArray
                    weights = jnp.exp(-weights)  # 使用softmax函数处理权重
                    weights /= jnp.sum(weights)  # 修改：使用jnp归一化权重
                    avg_params = jnp.average(jnp.stack(params_list, axis=0), axis=0, weights=weights)  # 新增：使用jnp计算权重平均
                    # avg_params = jnp.mean(jnp.stack(params_list, axis=0), axis=0)
                    for node in range(n_node - 1):
                        params_list[node] = avg_params
                    if b % 25 == 0:
                        avg_loss = jnp.mean(compute_loss(avg_params, x_test[:1024], y_test[:1024], k)[0])
                        loss_list.append(avg_loss)
                        acc_list.append(compute_accuracy(avg_params, x_test[:1024], y_test[:1024], k).mean())
                        tqdm.write(
                            f"world {world}, epoch {e}, batch {b}/{100}: loss {avg_loss}, accuracy {acc_list[-1]}")
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
        print(f'每个客户端的类别数：{n_class_nodes}')
        print(
            f'n_class {n_class}, test loss: {avg_test_loss}+-{std_test_loss}, test acc: {avg_test_acc}+-{std_test_acc}')
        class_test_loss.append(world_test_loss)
        class_test_acc.append(world_test_acc)
        class_train_acc.append(world_train_acc)
        class_train_loss.append(world_train_loss)

    os.makedirs(f'./{dataset}/qFedAvg-noniid-5-new/', exist_ok=True)
    jnp.save(f'./{dataset}/qFedAvg-noniid-5-new/test_loss.npy', class_test_loss)
    jnp.save(f'./{dataset}/qFedAvg-noniid-5-new/test_acc.npy', class_test_acc)
    jnp.save(f'./{dataset}/qFedAvg-noniid-5-new/train_acc.npy', class_train_acc)
    jnp.save(f'./{dataset}/qFedAvg-noniid-5-new/train_loss.npy', class_train_loss)
