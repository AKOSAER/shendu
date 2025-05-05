import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score
import matplotlib.pyplot as plt

# 加载MNIST数据集
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"].astype(np.uint8)

# 数据预处理
X = X.values.reshape(-1, 28, 28, 1) / 255.0  # 归一化并调整维度
y = np.eye(10)[y]  # one - hot编码

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


class LeNet5:
    def __init__(self):
        # 增加卷积核数量
        self.conv1 = np.random.randn(12, 1, 5, 5) * np.sqrt(1 / (5 * 5 * 1))
        self.conv2 = np.random.randn(32, 12, 5, 5) * np.sqrt(1 / (5 * 5 * 12))

        # 调整全连接层神经元数量
        self.fc1 = np.random.randn(32 * 4 * 4, 256) * np.sqrt(1 / (32 * 4 * 4))
        self.fc2 = np.random.randn(256, 128) * np.sqrt(1 / 256)
        self.fc3 = np.random.randn(128, 10) * np.sqrt(1 / 128)

        # 偏置
        self.b1 = np.zeros(12)
        self.b2 = np.zeros(32)
        self.b3 = np.zeros(256)
        self.b4 = np.zeros(128)
        self.b5 = np.zeros(10)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def conv2d(self, x, W, b, stride=1):
        """修正后的卷积实现"""
        batch, in_h, in_w, in_c = x.shape
        out_c, _, k_h, k_w = W.shape

        out_h = (in_h - k_h) // stride + 1
        out_w = (in_w - k_w) // stride + 1
        output = np.zeros((batch, out_h, out_w, out_c))

        # 调整卷积核维度为 (height, width, in_channels, out_channels)
        W_reshaped = W.transpose(2, 3, 1, 0)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                region = x[:, h_start:h_start + k_h, w_start:w_start + k_w, :]

                # 使用einsum代替tensordot
                output[:, i, j, :] = np.einsum(
                    'bhwc,hwco->bo', region, W_reshaped) + b
        return output

    def maxpool2d(self, x, pool_size=2):
        batch, h, w, c = x.shape
        out_h = h // pool_size
        out_w = w // pool_size

        output = np.zeros((batch, out_h, out_w, c))
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * pool_size
                w_start = j * pool_size
                region = x[:, h_start:h_start + pool_size,
                           w_start:w_start + pool_size, :]
                output[:, i, j, :] = np.max(region, axis=(1, 2))
        return output

    def forward(self, x):
        # 确保输入是4D (batch, height, width, channels)
        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=-1)

        # 第一层
        x = self.conv2d(x, self.conv1, self.b1)
        x = self.relu(x)
        x = self.maxpool2d(x)

        # 第二层
        x = self.conv2d(x, self.conv2, self.b2)
        x = self.relu(x)
        x = self.maxpool2d(x)

        # 全连接层
        x = x.reshape(x.shape[0], -1)
        x = self.relu(x @ self.fc1 + self.b3)
        x = self.relu(x @ self.fc2 + self.b4)
        x = x @ self.fc3 + self.b5

        return self.softmax(x)


class Trainer:
    def __init__(self, model, lr=0.01, reg_lambda=0.001):
        self.model = model
        self.lr = lr
        self.reg_lambda = reg_lambda

    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[np.arange(m), y_true.argmax(axis=1)])
        # 计算L2正则化项
        reg_loss = 0
        for param in [self.model.conv1, self.model.conv2, self.model.fc1, self.model.fc2, self.model.fc3]:
            reg_loss += np.sum(np.square(param))
        reg_loss = (self.reg_lambda / (2 * m)) * reg_loss
        return np.sum(log_likelihood) / m + reg_loss

    def train(self, X, y, epochs=150, batch_size=128):

        n_samples = X.shape[0]
        losses = []

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                X_batch, y_batch = X[batch_idx], y[batch_idx]

                # 前向传播
                output = self.model.forward(X_batch)
                loss = self.cross_entropy_loss(output, y_batch)

                # 反向传播（简化版）
                # 这里需要实现完整的梯度计算，为简洁起见省略具体实现
                # 实际使用时建议用自动微分库（如Autograd）或手动推导

                # 参数更新（示例）
                # self.model.conv1 -= self.lr * conv1_grad
                # ...

            losses.append(loss)
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
        return losses


def evaluate(model, X_test, y_test):
    predictions = model.forward(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    accuracy = accuracy_score(y_true, predictions)
    recall = recall_score(y_true, predictions, average='macro')
    rmse = np.sqrt(np.mean((y_true - predictions) ** 2))

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Recall: {recall * 100:.2f}%")
    print(f"Test RMSE: {rmse:.4f}")

    return accuracy, recall, rmse


# 初始化模型
lenet = LeNet5()
trainer = Trainer(lenet)

# 训练（实际使用时需补全反向传播）
losses = trainer.train(X_train, y_train)

# 评估
accuracy, recall, rmse = evaluate(lenet, X_test, y_test)

# 绘制损失曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 绘制评估指标
metrics = ['Accuracy', 'Recall', 'RMSE']
values = [accuracy, recall, rmse]
plt.subplot(1, 2, 2)
plt.bar(metrics, values)
plt.title('Evaluation Metrics')
plt.xlabel('Metrics')
plt.ylabel('Value')

plt.tight_layout()
plt.show()