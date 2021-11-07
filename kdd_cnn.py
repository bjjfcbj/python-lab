# %%
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from kdd_oh_set import ohkdd
from cnn_model import CNet

# %%
use_gpu = torch.cuda.is_available()
train_data = ohkdd(test_size=0, use_gpu=use_gpu, data_path='final_train.npy', return_type=2)
# valid_data = train_data.get_valid()
test_data = ohkdd(test_size=0, use_gpu=use_gpu, data_path='final_test.npy',
                  tar_path='kdd99_oh_label_corrected.npy', return_type=2)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
# valid_loader = DataLoader(valid_data, batch_size=100, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=200, shuffle=True, drop_last=True)

print("use_gpu:{}".format(use_gpu))

# %%
net = CNet()
loss_func = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-5)


# %%
def accuracy(net, input_loader):
    tp = tn = fp = fn = 0
    for x, y in input_loader:
        batch_x = Variable(x)
        batch_y = Variable(y)
        out = net(batch_x)
        res = torch.max(out, 1)[1]

        for i in range(0, len(batch_y)):
            if res[i] == batch_y[i]:
                if res[i] == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if batch_y[i] == 1:
                    fp += 1
                else:
                    fn += 1
    ac = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return [ac, precision, recall, f1]


# %%
loss_count = []
acc_count = []
if use_gpu:
    net = net.cuda()
    loss_func = loss_func.cuda()
print("training start...")
for epoch in range(3):
    for i, (x, y) in enumerate(train_loader):
        batch_x = Variable(x)
        batch_y = Variable(y)

        out = net(batch_x)  # 模型输出
        loss = loss_func(out, batch_y)  # 损失计算

        opt.zero_grad()  # 消除旧值
        loss.backward()  # 回传
        opt.step()  # 更新模型参数

        if i % 2000 == 0:  # 打点输出
            # loss_count.append(loss)
            print("{}th{}:\t".format(epoch + 1, i), loss.item())
            acc = accuracy(net, test_loader)
            acc_count.append(acc)
            print("acc:\t{}\nprecision:\t{}\nrecal:\t{}\nF1:\t{}".format(*acc))

    torch.save(net, r'kdd_cnn')  # 存储模型

# %%
plt.figure('cnn_acc')
acc_count = list(map(list, zip(*acc_count)))
plt.plot(acc_count[0], 'r', label='acc')
plt.plot(acc_count[1], 'b', label='precision')
plt.plot(acc_count[2], 'g', label='recall')
plt.plot(acc_count[3], 'y', label='f1')
plt.legend()
plt.ylim(0, 1.0)
plt.show()

# %%
acc = accuracy(net, test_loader)
acc_count.append(acc)
print("acc:\t{}\nprecision:\t{}\nrecal:\t{}\nF1:\t{}".format(*acc))

if __name__ == '__main__':
    print('model training end...')
