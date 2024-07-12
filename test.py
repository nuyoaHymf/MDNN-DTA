import torch
import torch.nn as nn
# temp = torch.load('/home/msyanmengfan/ymf/iedge-test/dataset/protein_feature/ABL1(F317I)p.pt')
# print(type(temp))

# protein_embedding = torch.load('/home/msyanmengfan/ymf/iedge-test/dataset/protein_feature/ABL1(F317I)p.pt')
# #print(protein_embedding['representations'][0].keys)


# emb = torch.load('/home/msyanmengfan/ymf/iedge-test/dataset/protein_feature/ABL1(F317I)p.pt')
# target_feature = emb["representations"].squeeze()
# target_global_feature = emb["sequence_repr"]
# print(type(target_global_feature))
# # target_size = len(target_feature)
# # target_edge_index = create_pseudo_graph(target_size, windows)

# input_size = 128  # 输入数据编码的维度
# hidden_size = 256  # 隐含层维度
# num_layers = 2     # 隐含层层数
# seq_length = 1000


# rnn = nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
# print("rnn：", rnn)
# x = torch.randn(1, 1000, 128)
# out = rnn(x)
# print("out:", out.shape)

# input_size = 128
# hidden_size = 256
# num_layers = 2
# seq_length = 1000
# batch_size = 1
# x = torch.randn(1, 128, 1000)
# # 创建 RNN 模型
# rnn = nn.RNN(input_size=seq_length, hidden_size=hidden_size, num_layers=num_layers)
# out = rnn(x)
# # 打印输出的形状
# print(type(out))

input_size = 100   # 输入数据编码的维度
hidden_size = 20   # 隐含层维度
num_layers = 4     # 隐含层层数

rnn = nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
print("rnn:",rnn)

seq_len = 10        # 句子长度
batch_size = 1      
x = torch.randn(seq_len,batch_size,input_size)        # 输入数据
h0 = torch.zeros(num_layers,batch_size,hidden_size)   # 输入数据

out, h = rnn(x, h0)  # 输出数据

print("out.shape:",out.shape)
print("h.shape:",h.shape)

