import tensorcircuit as tc
import optax
import jax.numpy as jnp
import jax
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from sklearn.mixture import GaussianMixture

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

# dataset = 'mnist'
dataset = 'fashion'
# 测量模式
readout_mode = 'softmax'
# readout_mode = 'sample'
# 编码模式
encoding_mode = 'vanilla'
# encoding_mode = 'mean'
# encoding_mode = 'half'

n = 8
n_node = 8
# 48层
k = 48


# 定义了一个量子电路函数
def clf(params, c, k):
    for j in range(k):
        for i in range(n - 1):
            c.cnot(i, i + 1)
        for i in range(n):
            # 这里为什么用3而不是其他的数字。这是因为每个模块中有三种不同类型的旋转门：RX，RZ 和 RX。
            # 因此，params 数组中每三行对应一个模块中的旋转门参数。3j 就是用来索引第 j 个模块中第一个 RX 门的参数。
            # 3j + 1 和 3j + 2 分别用来索引第 j 个模块中 RZ 和第二个 RX 的参数。
            c.rx(i, theta=params[3 * j, i])
            c.rz(i, theta=params[3 * j + 1, i])
            c.rx(i, theta=params[3 * j + 2, i])
    return c


# 定义了一个量子测量函数
# 根据给定的量子电路和测量模式，计算量子态的概率分布---从量子电路c中读
# 从量子电路读出的概率分布是用来探索量子电路 c 的状态和特性的，它可以用来分析量子纠缠、量子相干、量子噪声等现象，以及进行量子态重构、量子过程层析等技术
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


# 定义了一个量子损失函数
# 根据给定的参数、输入数据和标签，计算一个用于分类的量子电路的交叉熵损失
def loss(params, x, y, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return -jnp.mean(jnp.sum(y * jnp.log(probs + 1e-7), axis=-1))


loss = K.jit(loss, static_argnums=[3])


# 定义了一个量子准确率函数
def accuracy(params, x, y, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    # 使用 jnp.argmax 方法分别对 probs 和 y 求最大值所在的索引，即预测的类别和真实的类别，并比较它们是否相等；
    return jnp.argmax(probs, axis=-1) == jnp.argmax(y, axis=-1)


accuracy = K.jit(accuracy, static_argnums=[3])

# 三个 JAX 的变换函数来提高运行效率和简化代码
compute_loss = K.jit(K.vectorized_value_and_grad(loss, vectorized_argnums=[1, 2]), static_argnums=[3])
compute_accuracy = K.jit(K.vmap(accuracy, vectorized_argnums=[1, 2]), static_argnums=[3])


# 定义了一个量子预测函数
# 根据给定的参数和输入数据，计算一个用于分类的量子电路的输出概率分布----从输入数据中读
# 对输入数据进行预测的概率分布 probs 是用来评估量子分类器 clf 的性能和准确度的，它可以用来计算损失函数、准确率、混淆矩阵等指标，以及进行梯度下降、反向传播等优化算法
def pred(params, x, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return probs


pred = K.vmap(pred, vectorized_argnums=[1])

if __name__ == '__main__':
    # numpy data
    # 从 [MNIST] 或 [Fashion-MNIST] 数据集中加载数据，分为训练集和测试集。
    # 过滤掉类别为 8 或 9 的数据，只保留类别为 0 到 7 的数据。
    # 对数据进行归一化、去均值、缩放和重塑等操作，使其适合作为量子电路的输入。
    # 对类别标签进行 one-hot 编码，使其适合作为量子电路的输出。
    # 将训练集转换为一个 [TensorFlow Dataset] 对象，方便后续的批处理和迭代。
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
    if encoding_mode == 'vanilla':
        mean = 0
    elif encoding_mode == 'mean':
        mean = jnp.mean(x_train, axis=0)
    elif encoding_mode == 'half':
        mean = 0.5
    x_train = x_train - mean
    x_train = tf.image.resize(x_train[..., tf.newaxis], (int(2 ** (n / 2)), int(2 ** (n / 2)))).numpy()[..., 0].reshape(
        -1, 2 ** n)
    x_train = x_train / jnp.sqrt(jnp.sum(x_train ** 2, axis=-1, keepdims=True))

    x_test = x_test / 255.0
    x_test = x_test - mean
    x_test = tf.image.resize(x_test[..., tf.newaxis], (int(2 ** (n / 2)), int(2 ** (n / 2)))).numpy()[..., 0].reshape(
        -1, 2 ** n)
    x_test = x_test / jnp.sqrt(jnp.sum(x_test ** 2, axis=-1, keepdims=True))

    y_train = jax.nn.one_hot(y_train, n_node)
    y_test = jax.nn.one_hot(y_test, n_node)
    # 128个数据分为一批次
    data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)

    world_train_loss = []
    world_test_loss = []
    world_train_acc = []
    world_test_acc = []
    time_list = []
    # 10轮
    for world in tqdm(range(n_world)):
        # 记录开始时间
        start_time = time.time()
        key, subkey = jax.random.split(key)
        # 通过一个随机生成数，初始化一个参数矩阵，k是可调参数(旋转门的层数l=48)，n是量子比特的数量
        params = jax.random.normal(subkey, (3 * k, n))
        # 初始化一个优化器
        opt = optax.adam(learning_rate=1e-2)
        opt_state = opt.init(params)

        loss_list = []
        loss_list_train = []
        acc_list = []
        acc_list_train = []
        # 3个epoch
        for e in tqdm(range(3), leave=False):
            # 下标，数据，标签
            for i, (x, y) in enumerate(data):
                x = x.numpy()
                y = y.numpy()
                # 计算损失和梯度
                loss_val, grad_val = compute_loss(params, x, y, k)
                # 更新优化器状态，计算更新量
                updates, opt_state = opt.update(grad_val, opt_state, params)
                # 更新参数矩阵
                params = optax.apply_updates(params, updates)
                # 损失均值
                loss_mean = jnp.mean(loss_val)
                if i % 50 == 0:
                    acc = jnp.mean(compute_accuracy(params, x, y, k))
                    acc_list_train.append(acc)
                    loss_list_train.append(loss_mean)
                    loss_mean = jnp.mean(compute_loss(params, x_test[:1024], y_test[:1024], k)[0])
                    acc = jnp.mean(compute_accuracy(params, x_test[:1024], y_test[:1024], k))
                    acc_list.append(acc)
                    loss_list.append(loss_mean)
                    print(f'world {world}, epoch {e}, {i}/{len(data)}: loss={loss_mean:.4f}, acc={acc:.4f}')
        # 记录结束时间
        end_time = time.time()
        # 计算并打印通信时间
        communication_time = end_time - start_time
        print(f"Communication time: {communication_time} seconds")
        time_list.append(communication_time)
        # 测试精度与损失
        test_acc = jnp.mean(pred(params, x_test[:1024], k).argmax(axis=-1) == y_test[:1024].argmax(axis=-1))
        test_loss = -jnp.mean(jnp.log(pred(params, x_test[:1024], k)) * y_test[:1024])

        world_train_loss.append(loss_list)
        world_test_loss.append(test_loss)
        world_train_acc.append(acc_list)
        world_test_acc.append(test_acc)
        tqdm.write(f"world {world}: test loss {test_loss}, test accuracy {test_acc}")

    os.makedirs(f'./{dataset}/central1/', exist_ok=True)
    jnp.save(f'./{dataset}/central1/train_loss.npy', world_train_loss)
    jnp.save(f'./{dataset}/central1/train_acc.npy', world_train_acc)
    jnp.save(f'./{dataset}/central1/test_loss.npy', world_test_loss)
    jnp.save(f'./{dataset}/central1/test_acc.npy', world_test_acc)

    # 计算精度和损失的平均值和标准差
    avg_test_loss = jnp.mean(jnp.array(world_test_loss), axis=0)
    avg_test_acc = jnp.mean(jnp.array(world_test_acc), axis=0)
    std_test_loss = jnp.std(jnp.array(world_test_loss), axis=0)
    std_test_acc = jnp.std(jnp.array(world_test_acc), axis=0)
    # 输出每轮通信的时间戳，计算通信加速比
    print(time_list)
    print(f'test loss: {avg_test_loss}+-{std_test_loss}, test acc: {avg_test_acc}+-{std_test_acc}')
