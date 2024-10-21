import torch
import numpy as np
from deeprobust.graph.defense import GCNJaccard,SGC# GCN, RGCN,MedianGCN,#,GAT
from deeprobust.graph.defense import GCN as no_pyg_GCN
from deeprobust.graph.targeted_attack import SGAttack,Ours,Nettack,RND,FGA,IGAttack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset, Dpr2Pyg,Pyg2Dpr
from deeprobust.graph.defense_pyg import GCN,GAT,SAGE,AirGNN
import argparse
from tqdm import tqdm
import copy
import time
from util_class import my_eval
from scipy.sparse import csr_matrix, isspmatrix_csr, csr_array
from torch_geometric.datasets import CitationFull,Planetoid
from torch_geometric.datasets import Amazon
from scipy.sparse import lil_matrix

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

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# with open("./result.txt", "a") as file:
#     file.write(
#         f"\n\n|{'Dataset':^12s}|{'GNN_model':^12s}|{'Attack_model':^12s}|{'CLN_accuracy':^12s}|{'Attack_accuracy(poison)':^24s}|{'poison_overall_accuracy':^24s}|{'poison time':^24s}|{'poison Concealment':^24s}|{'Attack_accuracy(evasion)':^24s}|{'evasion_overall_accuracy':^24s}|{'evasion time':^24s}|{'evasion Concealment':^24s}|\n")


for data_name in ['pubmed']:#'pubmed' 'polblogs' 'acm', 'blogcatalog', 'uai', 'flickr','DBLP',Cora,Amazon
    for GNN_model_name in ['GAT']: #'GCN','GAT','SAGE','airgnn'
        for Attack_name in ['fga']: # 'rnd', 'Nettack','fga','IGAttack','SGAttack','Ours',
            #GNNs
            if GNN_model_name == 'GCN':
                GNN_model = GCN
            elif GNN_model_name == 'SAGE':
                GNN_model = SAGE
            elif GNN_model_name == 'GAT':
                GNN_model = GAT
            elif GNN_model_name == 'airgnn':
                GNN_model = AirGNN
            elif GNN_model_name == 'GCNJaccard':
                GNN_model = GCNJaccard

            #Attack model
            if Attack_name == 'SGAttack':
                Attack_model = SGAttack

            elif Attack_name == 'Nettack':
                Attack_model = Nettack

            elif Attack_name == 'rnd':
                Attack_model = RND

            elif Attack_name == 'fga':
                Attack_model = FGA

            elif Attack_name == 'Ours':
                Attack_model = Ours

            elif Attack_name == 'IGAttack':
                Attack_model = IGAttack


            # load Dataset

            #deeprobust dataset
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
            # else:
            #     data = CitationFull(root='./', name=data_name)
            # num_nodes = data[0].num_nodes
            # data = Pyg2Dpr(data)
            # adj, features, labels = data.adj, data.features, data.labels
            # # 生成随机索引
            # idx = torch.randperm(num_nodes)
            # # 60% 训练集，20% 验证集，20% 测试集
            # train_size = int(0.6 * num_nodes)
            # val_size = int(0.2 * num_nodes)
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

            #eval_clean
            args.dropout = 0.2
            args.lambda_amp = 0.5
            args.alpha = 0.1
            args.model = "AirGNN"
            model = GNN_model(nfeat=nfeat,
                              nhid=16,
                              nclass=nclass,
                              dropout=0.5,
                              # args=args,
                              device=device)

            model = model.to(device)

            model.fit(pyg_data, patience=30)

            model.test()
            output = model.predict()
            acc_test = accuracy(output[idx_test], labels[idx_test])
            CLN_accuracy=f'{acc_test.item():.4f}'

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

            if data_name == 'pubmed':
                adj = csr_matrix(adj)
                features = csr_matrix(features)
            else:
                adj = adj
                features=features

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

            eval = my_eval(GNN_model, Attack_model, adj, features, labels, idx_train, idx_val, idx_test,surrogate, pyg_data,nfeat,nclass)
            eval.reset(GNN_model, Attack_model, adj, features, labels, idx_train, idx_val, idx_test,surrogate, pyg_data)
            #evaluate
            # print(f'=== testing {GNN_model_name} on original(clean) graph ===')

            # eval.reset(GNN_model, Attack_model, adj, features, labels, idx_train, idx_val, idx_test,surrogate, pyg_data)
            # CLN_accuracy = eval.test()
            # print(f"CLN_accuracy:{CLN_accuracy}")
            # poison_time_start = time.time()
            # after_poison_accuracy =eval.multi_test_poison()
            # poison_time_end = time.time()
            # eval.reset(GNN_model, Attack_model, adj, features, labels, idx_train, idx_val, idx_test,surrogate, pyg_data)
            # poison_time = seconds_to_hms( poison_time_end -  poison_time_start)
            # with open("./result.txt", "a") as file:
            #     file.write(
            #         f"|{data_name:^12s}|{GNN_model_name:^12s}|{Attack_name:^12s}|{CLN_accuracy:^12s}|"
            #         f"{after_poison_accuracy:^24s}|"
            #         # f"{after_poison_overall_accuracy:^24s}|"
            #         f"{poison_time:^24s}|"
            #     )

            evasion_time_start = time.time()
            after_evasion_accuracy= eval.multi_test_evasion()
            evasion_time_end = time.time()
            eval.reset(GNN_model, Attack_model, adj, features, labels, idx_train, idx_val, idx_test,surrogate, pyg_data)
            evasion_time = seconds_to_hms(evasion_time_end - evasion_time_start)
            with open("./result.txt", "a") as file:
                file.write(
                    f"|{data_name:^12s}|{GNN_model_name:^12s}|{Attack_name:^12s}|{CLN_accuracy:^12s}|"
                    f"{after_poison_accuracy:^24s}|"
                    # f"{after_poison_overall_accuracy:^24s}|"
                    f"{poison_time:^24s}|"
                    f"{after_evasion_accuracy:^24s}|"
                    # f"{after_evasion_overall_accuracy}|"
                    f"{evasion_time:^24s}|\n"
                )
            del data, adj, features, labels, idx_train, idx_val, idx_test,pyg_data,idx_unlabeled,model,eval
