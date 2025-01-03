import tensorcircuit as tc
import optax
import jax.numpy as jnp
import jax
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

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

# 外循环10轮
n_world = 10

# dataset = 'mnist'
dataset = 'fashion'
readout_mode = 'softmax'
# readout_mode = 'sample'
# 编码方式
encoding_mode = 'vanilla'
# encoding_mode = 'mean'
# encoding_mode = 'half'

# 比特数
n = 8
# 客户端数
n_node = 8
k = 48


# 它接受三个参数 x、y 和 class_list。x 和 y 分别是输入特征和标签，class_list 是一个包含要保留的标签类别的整数列表。
# 函数的目的是从输入特征和标签中过滤出位于 class_list 中的标签类别，并将它们转换为 one-hot 编码的形式。
# 类别数太多了，只需要过滤出自己想要的就行
def filter(x, y, class_list):
    keep = jnp.zeros(len(y)).astype(bool)
    for c in class_list:
        keep = keep | (y == c)
    x, y = x[keep], y[keep]
    y = jax.nn.one_hot(y, n_node)
    return x, y


# 量子线路
def clf(params, c, k):
    for j in range(k):
        for i in range(n - 1):
            c.cnot(i, i + 1)
        for i in range(n):
            c.rx(i, theta=params[3 * j, i])
            c.rz(i, theta=params[3 * j + 1, i])
            c.rx(i, theta=params[3 * j + 2, i])
    return c


# 输出概率分布
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


# 损失函数
def loss(params, x, y, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return -jnp.mean(jnp.sum(y * jnp.log(probs + 1e-7), axis=-1))


loss = K.jit(loss, static_argnums=[3])


# 准确率
def accuracy(params, x, y, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return jnp.argmax(probs, axis=-1) == jnp.argmax(y, axis=-1)


accuracy = K.jit(accuracy, static_argnums=[3])

compute_loss = K.jit(K.vectorized_value_and_grad(loss, vectorized_argnums=[1, 2]), static_argnums=[3])
compute_accuracy = K.jit(K.vmap(accuracy, vectorized_argnums=[1, 2]), static_argnums=[3])


# 预测值
def pred(params, x, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return probs


pred = K.vmap(pred, vectorized_argnums=[1])

if __name__ == '__main__':
    # numpy data
    # 加载对应的数据
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'fashion':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # 除去类别8和9
    ind = y_test == 9
    x_test, y_test = x_test[~ind], y_test[~ind]
    ind = y_test == 8
    x_test, y_test = x_test[~ind], y_test[~ind]
    ind = y_train == 9
    x_train, y_train = x_train[~ind], y_train[~ind]
    ind = y_train == 8
    x_train, y_train = x_train[~ind], y_train[~ind]

    x_train = x_train / 255.0
    # 量子编码
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

    # 测试集标签变成独热向量------不同点，也没有分批次打包数据
    y_test = jax.nn.one_hot(y_test, n_node)

    world_train_loss = []
    world_test_loss = []
    world_train_acc = []
    world_test_acc = []

    for world in tqdm(range(n_world)):

        params_list = []
        opt_state_list = []
        data_list = []
        iter_list = []
        # 准备好数据，参数矩阵，优化器
        for node in range(n_node - 1):
            # 先过滤出自己想要的数据集类别数（类别0和当前节点），每个客户端只进行二分类，星型结构(non-iid结构，数据偏移最严重)
            x_train_node, y_train_node = filter(x_train, y_train, [0, node + 1])
            # x_train_node, y_train_node = x_train, jax.nn.one_hot(y_train, n_node)
            # 将过滤后的数据集进行打包，批大小128，一次取128个数据进行处理
            data = tf.data.Dataset.from_tensor_slices((x_train_node, y_train_node)).batch(128)
            # 放进数据列表中
            data_list.append(data)
            # 将生成的 data 数据集构造一个 Python 迭代器对象 iter(data)，并将其添加到 iter_list 列表中。
            # 在训练过程中，将需要反复从 data 数据集中提取数据进行训练，而 iter_list 中存储的每个 Python 迭代器对象可以通过 next 函数进行迭代，
            # 得到数据集中的一个批次数据，进而对网络进行训练。可以通过next函数进行访问
            iter_list.append(iter(data))

            key, subkey = jax.random.split(key)
            # 通过随机数，生成参数矩阵
            params = jax.random.normal(subkey, (3 * k, n))
            # 设置优化器，并初始化
            opt = optax.adam(learning_rate=1e-2)
            opt_state = opt.init(params)
            # 参数列表，加载并更新参数矩阵
            params_list.append(params)
            # 优化器状态列表
            opt_state_list.append(opt_state)

        loss_list = []
        acc_list = []
        # 训练5次
        for e in tqdm(range(5), leave=False):
            # 每次使用100个批次来训练本地模型，一共5次
            for b in range(100):
                # 8个客户端，客户端训练，每个客户端取数据[0，node+1]
                for node in range(n_node - 1):
                    try:
                        # 从迭代列表中获取下一批次的数据
                        x, y = next(iter_list[node])
                    except StopIteration:
                        # 如果为空，就重新开始
                        iter_list[node] = iter(data_list[node])
                        x, y = next(iter_list[node])
                    # 将数据转成numpy格式
                    x = x.numpy()
                    y = y.numpy()
                    # 计算损失和梯度
                    loss_val, grad_val = compute_loss(params_list[node], x, y, k)
                    # 更新优化器状态和参数矩阵
                    updates, opt_state_list[node] = opt.update(grad_val, opt_state_list[node], params_list[node])
                    params_list[node] = optax.apply_updates(params_list[node], updates)

                # 服务器端的工作，参数聚合
                # 将所有节点的参数矩阵求平均值
                # 并将平均值作为所有结点的参数矩阵值
                avg_params = jnp.mean(jnp.stack(params_list, axis=0), axis=0)
                for node in range(n_node - 1):
                    params_list[node] = avg_params

                # 计算对应的损失和精度
                if b % 25 == 0:
                    # 取测试集前1024个样本进行计算
                    avg_loss = jnp.mean(compute_loss(avg_params, x_test[:1024], y_test[:1024], k)[0])
                    loss_list.append(avg_loss)
                    acc_list.append(compute_accuracy(avg_params, x_test[:1024], y_test[:1024], k).mean())
                    tqdm.write(f"world {world}, epoch {e}, batch {b}/{100}: loss {avg_loss}, accuracy {acc_list[-1]}")

        # 测试精度和损失
        # 返回概率最大的类别和真实类别进行比较
        test_acc = jnp.mean(pred(avg_params, x_test[:1024], k).argmax(axis=-1) == y_test[:1024].argmax(axis=-1))
        test_loss = -jnp.mean(jnp.log(pred(avg_params, x_test[:1024], k)) * y_test[:1024])

        world_train_loss.append(loss_list)
        world_test_loss.append(test_loss)
        world_train_acc.append(acc_list)
        world_test_acc.append(test_acc)
        tqdm.write(f"world {world}: test loss {test_loss}, test accuracy {test_acc}")

    # 将数据写入文件
    os.makedirs(f'./{dataset}/qFedAvg-2/', exist_ok=True)
    jnp.save(f'./{dataset}/qFedAvg-2/train_loss.npy', world_train_loss)
    jnp.save(f'./{dataset}/qFedAvg-2/train_acc.npy', world_train_acc)
    jnp.save(f'./{dataset}/qFedAvg-2/test_loss.npy', world_test_loss)
    jnp.save(f'./{dataset}/qFedAvg-2/test_acc.npy', world_test_acc)

    # 测试损失和精度的均值和标准差
    avg_test_loss = jnp.mean(jnp.array(world_test_loss), axis=0)
    avg_test_acc = jnp.mean(jnp.array(world_test_acc), axis=0)
    std_test_loss = jnp.std(jnp.array(world_test_loss), axis=0)
    std_test_acc = jnp.std(jnp.array(world_test_acc), axis=0)
    print(f'test loss: {avg_test_loss}+-{std_test_loss}, test acc: {avg_test_acc}+-{std_test_acc}')
