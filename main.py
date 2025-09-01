import torch
from deeprobust.graph.defense import GCNJaccard,SGC# GCN, RGCN,MedianGCN,#,GAT
from deeprobust.graph.defense import GCN as no_pyg_GCN
from deeprobust.graph.targeted_attack import SGAttack,Ours,Nettack,RND,FGA,IGAttack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset, Dpr2Pyg,Pyg2Dpr
from deeprobust.graph.defense_pyg import GCN,GAT,SAGE,AirGNN
import argparse
from tqdm import tqdm
import os
import copy
import time
from util_class import my_eval
from scipy.sparse import csr_matrix, isspmatrix_csr, csr_array
from torch_geometric.datasets import CitationFull, Planetoid, Reddit
from torch_geometric.datasets import Amazon
from torch_geometric.utils import degree, subgraph
from torch_geometric.data import Data
from scipy.sparse import lil_matrix
def sample_high_degree_subgraph(data, num_samples=300):
    """
    基于度数采样节点并构建子图。

    参数:
    - data: PyG 数据对象
    - num_samples: 要采样的节点数量

    返回:
    - Data: 包含采样节点的子图，格式为 PyG 数据对象
    """
    # 计算每个节点的度数
    node_degrees = degree(data.edge_index[0], num_nodes=data.x.shape[0])

    # 找到度数最高的前 num_samples 个节点
    top_nodes = torch.topk(node_degrees, num_samples).indices

    # 使用采样节点构建子图
    subgraph_edge_index, subgraph_edge_attr = subgraph(top_nodes, data.edge_index, relabel_nodes=True)

    # 计算子图中每个节点的度数
    subgraph_degrees = degree(subgraph_edge_index[0], num_nodes=num_samples)

    # 过滤掉度为 0 的节点
    valid_nodes = subgraph_degrees > 0
    valid_node_indices = top_nodes[valid_nodes]

    # 重新生成子图，确保没有度为 0 的节点
    subgraph_edge_index, subgraph_edge_attr = subgraph(valid_node_indices, data.edge_index, relabel_nodes=True)

    # 构建新的 PyG 数据对象
    subgraph_data = Data(
        x=data.x[valid_node_indices],  # 采样节点的特征
        edge_index=subgraph_edge_index,  # 子图的边索引
        edge_attr=subgraph_edge_attr,  # 子图的边属性（如果有）
    )

    # 如果数据中包含节点标签，将其添加到子图数据中
    if hasattr(data, 'y'):
        subgraph_data.y = data.y[valid_node_indices]

    return subgraph_data

def seconds_to_hms(time_seconds):
    hours = time_seconds // 3600
    minutes = (time_seconds % 3600) // 60
    seconds = time_seconds % 60
    return f"{hours:.0f}h {minutes:.0f}m {seconds:.2f}s"

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05, help='pertubation rate')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# with open("./result-2025-1.txt", "a") as file:
#     file.write(
#         f"\n\n|{'Dataset':^12s}|{'GNN_model':^12s}|{'Attack_model':^12s}|{'CLN_accuracy':^12s}|{'Attack_accuracy(poison)':^24s}|{'poison_overall_accuracy':^24s}|{'poison time':^24s}|{'poison Concealment':^24s}|{'Attack_accuracy(evasion)':^24s}|{'evasion_overall_accuracy':^24s}|{'evasion time':^24s}|{'evasion Concealment':^24s}|\n")


for data_name in ['cora','citeseer','polblogs']:# xxx'pubmed' citeseer,'DBLP',cora,Amazon,Reddit,polblogs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # load Dataset
    # deeprobust dataset
    data = Dataset(root='./', name=data_name)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    nfeat = features.shape[1]
    nclass = labels.max().item() + 1
    pyg_data = Dpr2Pyg(data)

    # #pyg dataset
    # if data_name == 'Amazon':
    #     data = Amazon(root='./',name='Computers')
    # elif data_name == 'Reddit':
    #     data = Reddit(root='./')
    # else:
    #     data = CitationFull(root='./', name=data_name)
    #     # data = Planetoid(root='./', name=data_name)
    # data=data[0]
    # num_nodes = data.x.shape[0]
    # data = Pyg2Dpr(data)
    # adj, features, labels = data.adj, data.features, data.labels
    # # 生成随机索引
    # idx = torch.randperm(num_nodes)
    # # 10% 训练集，10% 验证集，80% 测试集
    # train_size = int(0.1 * num_nodes)
    # val_size = int(0.1 * num_nodes)
    # test_size = num_nodes - train_size - val_size
    # # 划分索引
    # idx_train = idx[:train_size]
    # idx_val = idx[train_size:train_size + val_size]
    # idx_test = idx[train_size + val_size:]
    # idx_unlabeled = np.union1d(idx_val, idx_test)
    # data.idx_train, data.idx_val, data.idx_test=idx_train, idx_val, idx_test
    # nfeat = features.shape[1]
    # nclass = labels.max().item() + 1
    # features = csr_matrix(features)
    # features = features.astype(np.float32)
    # pyg_data = Dpr2Pyg(data)

    if data_name == 'pubmed':
        adj = csr_matrix(adj)
        features = csr_matrix(features)
    else:
        adj = adj
        features = features

    for GNN_model_name in ['GCN','GAT','SAGE']: #'GCN','GAT','SAGE'
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        # GNNs
        if GNN_model_name == 'GCN':
            GNN_model = GCN
            cos_ppr_alpha = 0.1
        elif GNN_model_name == 'SAGE':
            GNN_model = SAGE
            cos_ppr_alpha = 0.9
        elif GNN_model_name == 'GAT':
            GNN_model = GAT
            cos_ppr_alpha = 0.1


        # eval_clean
        model = GNN_model(nfeat=nfeat,
                          nhid=16,
                          nclass=nclass,
                          dropout=0.5,
                          device=device)

        model = model.to(device)
        model.fit(pyg_data, patience=30)
        # model.test()
        output = model.predict()
        if os.path.exists(f"{data_name}_node_list.pt"):
            node_list = torch.load(f"{data_name}_node_list.pt")
        else:
            margin_dict = {}
            for idx in idx_test:
                margin = classification_margin(output[idx], labels[idx])
                if margin < 0:  # only keep the nodes correctly classified
                    continue
                margin_dict[idx] = margin
            sorted_margins = sorted(margin_dict.items(), key=lambda x: x[1], reverse=True)
            high = [x for x, y in sorted_margins[: 125]]
            low = [x for x, y in sorted_margins[-125:]]
            other = [x for x, y in sorted_margins[125: -125]]
            other = np.random.choice(other, 250, replace=False).tolist()
            node_list = high + low + other
            torch.save(node_list,f"{data_name}_node_list.pt")

        for Attack_name in ['IGAttack']: # 'rnd', 'Nettack','fga','SGAttack','Ours',
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            #Attack model
            if Attack_name == 'SGAttack':
                Attack_model = SGAttack

            elif Attack_name == 'Nettack':
                Attack_model = Nettack

            elif Attack_name == 'rnd':
                Attack_model = RND

            elif Attack_name == 'fga':
                Attack_model = FGA

            elif Attack_name == 'IGAttack':
                Attack_model = IGAttack

            elif Attack_name == 'Ours':
                Attack_model = Ours


            # surrogate model：
            if Attack_name in ['SGAttack', 'Ours']:
                surrogate = SGC(nfeat=nfeat,
                                nclass=nclass, K=2,
                                lr=0.01, device=device).to(device)

                surrogate.fit(pyg_data, verbose=False)  # train with earlystopping
                # surrogate.test()
            else:
                surrogate = no_pyg_GCN(nfeat=nfeat, nclass=nclass,
                                nhid=16, device=device)

                surrogate = surrogate.to(device)
                surrogate.fit(features, adj, labels, idx_train, idx_val)

            # surrogate = SGC(nfeat=nfeat,
            #                 nclass=nclass, K=2,
            #                 lr=0.01, device=device).to(device)
            # surrogate.fit(pyg_data, verbose=False)  # train with earlystopping
            # surrogate.test()

            # Attack
            # attack_one_time_start = time.time()
            # target_node = 0
            # assert target_node in idx_unlabeled
            # if Attack_model == Ours:
            #     model = Attack_model(adj,surrogate, attack_structure=True, attack_features=False, device=device)
            # elif Attack_model == Nettack:
            #     model = Attack_model(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
            # elif Attack_model == FGA:
            #     model = FGA(surrogate, nnodes=adj.shape[0], device=device)
            # elif Attack_model == RND:
            #     model = RND()
            # elif Attack_model == SGAttack:
            #     model = Attack_model(surrogate, attack_structure=True, attack_features=False, device=device)
            # model = model.to(device)

            # degrees = adj.sum(0).A1
            # n_perturbations = int(degrees[target_node])
            #
            # if Attack_name == 'fga':
            #     model.attack(features,adj, labels,idx_train, target_node, n_perturbations)  # FGA
            # elif Attack_name == 'rnd':
            #     model.attack(adj, labels, idx_train, target_node, n_perturbations)
            # else:
            #     model.attack(features, adj, labels, target_node, n_perturbations, direct=True)
            # attack_one_time_end = time.time()


            cos_ppr_top_n = 1000
            ppr_p = 0.85
            eval = my_eval(GNN_model, Attack_model, adj, features, labels, idx_train, idx_val, idx_test,node_list,surrogate, pyg_data,nfeat,nclass,cos_ppr_alpha=cos_ppr_alpha,cos_ppr_top_n=cos_ppr_top_n,ppr_p=ppr_p,)
            eval.reset(GNN_model, Attack_model, adj, features, labels, idx_train, idx_val, idx_test,surrogate, pyg_data)
            #evaluate

            eval.reset(GNN_model, Attack_model, adj, features, labels, idx_train, idx_val, idx_test,surrogate, pyg_data)
            CLN_accuracy = eval.test(model=model)
            print(f"CLN_accuracy:{CLN_accuracy}")

            # poison_time_start = time.time()
            # after_poison_accuracy =eval.multi_test_poison()
            # poison_time_end = time.time()
            # eval.reset(GNN_model, Attack_model, adj, features, labels, idx_train, idx_val, idx_test,surrogate, pyg_data)
            # poison_time = seconds_to_hms( poison_time_end -  poison_time_start)
            with open("./result-2025-1.txt", "a") as file:
                file.write(
                    f"|{data_name:^12s}|{GNN_model_name:^12s}|{Attack_name:^12s}|{CLN_accuracy:^12s}|"
                    # f"{after_poison_accuracy:^24s}|"
                    # # f"{after_poison_overall_accuracy:^24s}|"
                    # f"{poison_time:^24s}|"
                )

            evasion_time_start = time.time()
            after_evasion_accuracy= eval.multi_test_evasion()
            evasion_time_end = time.time()
            eval.reset(GNN_model, Attack_model, adj, features, labels, idx_train, idx_val, idx_test,surrogate, pyg_data)
            evasion_time = seconds_to_hms(evasion_time_end - evasion_time_start)
            with open("./result-2025-1.txt", "a") as file:
                file.write(
                    f"{after_evasion_accuracy:^24s}|"
                    # f"{after_evasion_overall_accuracy}|"
                    f"{evasion_time:^24s}|\n"
                )
