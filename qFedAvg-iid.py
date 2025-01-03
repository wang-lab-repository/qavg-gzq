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
# 设置随机种子
key = jax.random.PRNGKey(42)
tf.random.set_seed(42)

n_world = 10

# dataset = 'mnist'
dataset = 'fashion'
# dataset = 'shirt'
readout_mode = 'softmax'
# readout_mode = 'sample'
encoding_mode = 'vanilla'
# encoding_mode = 'mean'
# encoding_mode = 'half'
# encoding_mode = 'amplitude'
# encoding_mode = 'angle'

# 量子比特数
n = 8
# 客户端数
n_node = 8
# 量子电路层数
k = 48


# 过滤出想要的类别
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
            # 这里为什么用3而不是其他的数字。这是因为每个模块中有三种不同类型的旋转门：RX，RZ 和 RX。
            # 因此，params 数组中每三行对应一个模块中的旋转门参数。3j 就是用来索引第 j 个模块中第一个 RX 门的参数。
            # 3j + 1 和 3j + 2 分别用来索引第 j 个模块中 RZ 和第二个 RX 的参数。
            c.rx(i, theta=params[3 * j, i])
            c.rz(i, theta=params[3 * j + 1, i])
            c.rx(i, theta=params[3 * j + 2, i])

    return c


# 量子分类器
def clf1(params, c, k):
    for j in range(k):
        for i in range(n - 1):
            c.cnot(i, i + 1)
            c.swap(i, i + 1)
            for i in range(n):
                c.h(i)
                c.s(i)
                c.t(i)
                c.rx(i, theta=params[4 * j, i])
                c.rz(i, theta=params[4 * j + 1, i])
                c.rx(i, theta=params[4 * j + 2, i])
                c.rz(i, theta=params[4 * j + 3, i])
    return c


# 通过量子线路中读出概率分布
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


# 计算损失---交叉熵损失
def loss(params, x, y, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    reg = 1e-4  # 正则化系数
    return -jnp.mean(jnp.sum(y * jnp.log(probs + 1e-7), axis=-1)) + reg * jnp.sum(params ** 2)
    # return -jnp.mean(jnp.sum(y * jnp.log(probs + 1e-7), axis=-1))


loss = K.jit(loss, static_argnums=[3])


# 计算精度
def accuracy(params, x, y, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return jnp.argmax(probs, axis=-1) == jnp.argmax(y, axis=-1)


accuracy = K.jit(accuracy, static_argnums=[3])

compute_loss = K.jit(K.vectorized_value_and_grad(loss, vectorized_argnums=[1, 2]), static_argnums=[3])
compute_accuracy = K.jit(K.vmap(accuracy, vectorized_argnums=[1, 2]), static_argnums=[3])


# 通过输入数据x获取概率分布
def pred(params, x, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return probs


pred = K.vmap(pred, vectorized_argnums=[1])

if __name__ == '__main__':
    # numpy data
    # 首先过滤掉了类别为8或者9的样本，然后把图像归一化到0到1之间，然后根据不同的编码模式，把图像减去一个均值，
    # 然后把图像缩放到n位的量子寄存器的大小，然后把图像归一化到单位长度，最后把标签转换成one-hot编码的形式。
    # 这些步骤都是为了让你的数据集适合用于量子机器学习的任务。
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'fashion':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset == 'shirt':
        # 加载fashion_mnist数据集
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        # 将多分类问题转换为二分类问题：是否为衬衫（类别标签为6）
        y_train = jnp.where(y_train == 6, 1, 0)
        y_test = jnp.where(y_test == 6, 1, 0)
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
    # n = 8，根据客户端的数量动态改变输入图片的大小，将图片大小转成16X16形状，最终编码成8量子比特
    x_train = tf.image.resize(x_train[..., tf.newaxis], (int(2 ** (n / 2)), int(2 ** (n / 2)))).numpy()[..., 0].reshape(
        -1, 2 ** n)
    x_train = x_train / jnp.sqrt(jnp.sum(x_train ** 2, axis=-1, keepdims=True))

    x_test = x_test / 255.0
    x_test = x_test - mean
    x_test = tf.image.resize(x_test[..., tf.newaxis], (int(2 ** (n / 2)), int(2 ** (n / 2)))).numpy()[..., 0].reshape(
        -1, 2 ** n)
    x_test = x_test / jnp.sqrt(jnp.sum(x_test ** 2, axis=-1, keepdims=True))
    y_test = jax.nn.one_hot(y_test, n_node)

    world_train_loss = []
    world_test_loss = []
    world_train_acc = []
    world_test_acc = []
    time_list = []
    for world in tqdm(range(n_world)):

        params_list = []
        opt_state_list = []
        data_list = []
        iter_list = []

        # 记录开始时间
        start_time = time.time()
        for node in range(n_node - 1):
            # 包含10个标签，属于iid的情况
            # 数据集转换成一个批次生成器。然后你为每个节点初始化一个随机的参数矩阵，以及一个优化器和一个优化状态
            x_train_node, y_train_node = filter(x_train, y_train, range(10))
            # x_train_node, y_train_node = x_train, jax.nn.one_hot(y_train, n_node)
            data = tf.data.Dataset.from_tensor_slices((x_train_node, y_train_node)).batch(128)
            data_list.append(data)
            # 将生成的 data 数据集构造一个 Python 迭代器对象 iter(data)，并将其添加到 iter_list 列表中。
            # 在训练过程中，将需要反复从 data 数据集中提取数据进行训练，而 iter_list 中存储的每个 Python 迭代器对象可以通过 next 函数进行迭代，
            # 得到数据集中的一个批次数据，进而对网络进行训练。可以通过next函数进行访问
            iter_list.append(iter(data))

            key, subkey = jax.random.split(key)
            params = jax.random.normal(subkey, (3 * k, n))
            opt = optax.adam(learning_rate=1e-2)
            opt_state = opt.init(params)
            params_list.append(params)
            opt_state_list.append(opt_state)

        loss_list = []
        acc_list = []
        # 5个epoch，每个epoch有100batch
        for e in tqdm(range(5), leave=False):
            for b in range(100):
                # 对于每个客户端，获取一个批次的数据，计算损失，更新参数矩阵和优化器
                for node in range(n_node - 1):
                    try:
                        # 获取这个批次的数据
                        x, y = next(iter_list[node])
                    except StopIteration:
                        # 如果没有数据，便从头开始
                        iter_list[node] = iter(data_list[node])
                        x, y = next(iter_list[node])
                    x = x.numpy()
                    y = y.numpy()
                    loss_val, grad_val = compute_loss(params_list[node], x, y, k)
                    updates, opt_state_list[node] = opt.update(grad_val, opt_state_list[node], params_list[node])
                    params_list[node] = optax.apply_updates(params_list[node], updates)

                # 待所有的客户端都计算完后
                # 中心服务器将所有节点的参数矩阵求平均值，并将平均值作为所有结点的参数矩阵值
                avg_params = jnp.mean(jnp.stack(params_list, axis=0), axis=0)
                for node in range(n_node - 1):
                    params_list[node] = avg_params
                # 输出训练损失和精度
                if b % 25 == 0:
                    avg_loss = jnp.mean(compute_loss(avg_params, x_test[:1024], y_test[:1024], k)[0])
                    loss_list.append(avg_loss)
                    acc_list.append(compute_accuracy(avg_params, x_test[:1024], y_test[:1024], k).mean())
                    tqdm.write(f"world {world}, epoch {e}, batch {b}/{100}: loss {avg_loss}, accuracy {acc_list[-1]}")

        # 记录结束时间
        end_time = time.time()
        # 计算并打印通信时间
        communication_time = end_time - start_time
        print(f"Communication time: {communication_time} seconds")
        time_list.append(communication_time)
        # 计算测试损失和精度
        test_acc = jnp.mean(pred(avg_params, x_test[:1024], k).argmax(axis=-1) == y_test[:1024].argmax(axis=-1))
        test_loss = -jnp.mean(jnp.log(pred(avg_params, x_test[:1024], k)) * y_test[:1024])

        world_train_loss.append(loss_list)
        world_test_loss.append(test_loss)
        world_train_acc.append(acc_list)
        world_test_acc.append(test_acc)
        tqdm.write(f"world {world}: test loss {test_loss}, test accuracy {test_acc}")

    os.makedirs(f'./{dataset}/qFedAvg/', exist_ok=True)
    jnp.save(f'./{dataset}/qFedAvg/train_loss.npy', world_train_loss)
    jnp.save(f'./{dataset}/qFedAvg/train_acc.npy', world_train_acc)
    jnp.save(f'./{dataset}/qFedAvg/test_loss.npy', world_test_loss)
    jnp.save(f'./{dataset}/qFedAvg/test_acc.npy', world_test_acc)

    avg_test_loss = jnp.mean(jnp.array(world_test_loss), axis=0)
    avg_test_acc = jnp.mean(jnp.array(world_test_acc), axis=0)
    std_test_loss = jnp.std(jnp.array(world_test_loss), axis=0)
    std_test_acc = jnp.std(jnp.array(world_test_acc), axis=0)
    # 输出每轮通信的时间戳，计算通信加速比
    print(time_list)
    print(f'test loss: {avg_test_loss}+-{std_test_loss}, test acc: {avg_test_acc}+-{std_test_acc}')
