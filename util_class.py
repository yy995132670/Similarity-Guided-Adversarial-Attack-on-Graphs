from deeprobust.graph.utils import *
from deeprobust.graph.targeted_attack import SGAttack,Nettack,RND,FGA,Ours,IGAttack
from deeprobust.graph.defense import GCNJaccard,SGC#GCN, SGC,RGCN,MedianGCN,GCNJaccard#GAT,
from copy import deepcopy
import argparse
import torch
import copy
from tqdm import tqdm
from deeprobust.graph.defense_pyg import GCN,GAT,SAGE,AirGNN
from torch_geometric.utils import degree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.dropout = 0.2
args.lambda_amp = 0.5
args.alpha = 0.1
args.model = "AirGNN"
class my_eval:
    def __init__(self, GNN_model, Attack_model,adj, features, labels, idx_train, idx_val, idx_test, surrogate,pyg_data,nfeat,nclass):
        self.GNN_model=GNN_model
        self.Attack_model=Attack_model
        self.adj=adj
        self.features=features
        self.labels=labels
        self.idx_train=idx_train
        self.idx_val=idx_val
        self.idx_test=idx_test
        self.surrogate=surrogate
        self.pyg_data=pyg_data
        self.clean_data=pyg_data
        self.final_adj=adj
        self.nfeat=nfeat
        self.nclass=nclass

    def reset(self, GNN_model, Attack_model,adj, features, labels, idx_train, idx_val, idx_test, surrogate,pyg_data):
        self.GNN_model=GNN_model
        self.Attack_model=Attack_model
        self.adj=adj
        self.features=features
        self.labels=labels
        self.idx_train=idx_train
        self.idx_val=idx_val
        self.idx_test=idx_test
        self.surrogate=surrogate
        self.pyg_data=pyg_data
        self.clean_data = pyg_data
        self.final_adj=adj


    def get_final_adj(self,modified_adj):

        # # 对比 modified_adj 和 adj
        # for i in range(self.adj.shape[0]):
        #     for j in range(self.adj.shape[1]):
        #         # 如果 modified_adj 和 adj 不一样，说明这里发生了变化
        #         if modified_adj[i, j] != self.adj[i, j]:
        #             # 将 modified_adj 的状态累积到 final_adj
        #             self.final_adj[i, j] = modified_adj[i, j]

        modified_adj=torch.tensor(modified_adj.toarray())
        adj = torch.tensor(self.adj.toarray())
        final_adj = torch.tensor(self.final_adj.toarray())
        final_adj = torch.where(modified_adj != adj, modified_adj, final_adj)

    def test(self):
        cnt=0
        node_list = self.select_nodes()
        num = len(node_list)
        for target_node in tqdm(node_list):
            acc = self.single_test(self.adj.toarray(), self.features.toarray(), target_node)
            if acc == 0:
                cnt += 1
        clean_accuracy = 1 - cnt / num
        return f"{clean_accuracy:.4f}"


    def select_nodes(self, target_GNN_model=None):
        '''
        selecting nodes as reported in Nettack paper:
        (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
        (ii) the 10 nodes with lowest margin (but still correctly classified) and
        (iii) 20 more nodes randomly
        '''

        if target_GNN_model is None:
            target_GNN_model = GCN(nfeat= self.nfeat,
                                  nhid=16,
                                  nclass= self.nclass,
                                  dropout=0.5, device=device)

            target_GNN_model = target_GNN_model.to(device)
            target_GNN_model.fit(self.pyg_data)
        # target_GNN_model.test()
        output = target_GNN_model.predict()

        margin_dict = {}
        for idx in  self.idx_test:
            margin = classification_margin(output[idx],  self.labels[idx])
            if margin < 0:  # only keep the nodes correctly classified
                continue
            margin_dict[idx] = margin
        sorted_margins = sorted(margin_dict.items(), key=lambda x: x[1], reverse=True)
        high = [x for x, y in sorted_margins[: 10]]
        low = [x for x, y in sorted_margins[-10:]]
        other = [x for x, y in sorted_margins[10: -10]]
        other = np.random.choice(other, 480, replace=False).tolist()

        return high + low + other


    def multi_test_poison(self):
        cnt = 0
        degrees =  self.adj.sum(0).A1
        node_list =  self.select_nodes()
        num = len(node_list)
        print('=== [Poisoning] Attacking %s nodes respectively ===' % num)
        for target_node in tqdm(node_list):
            if isinstance(target_node, torch.Tensor):
                target_node = target_node.item()
            n_perturbations = int(degrees[target_node])

            if  self.Attack_model == Ours:
                model =  self.Attack_model( self.adj,  self.surrogate, attack_structure=True, attack_features=False, device=device)
            elif  self.Attack_model in [Nettack,IGAttack]:
                model =  self.Attack_model( self.surrogate, nnodes= self.adj.shape[0], attack_structure=True, attack_features=False,device=device)
            elif  self.Attack_model == FGA:
                model = FGA( self.surrogate, nnodes= self.adj.shape[0], device=device)
            elif  self.Attack_model == RND:
                model = RND()
            elif  self.Attack_model == SGAttack:
                model =  self.Attack_model( self.surrogate, attack_structure=True, attack_features=False, device=device)

            model = model.to(device)
            if self.Attack_model in [FGA, IGAttack]:
                model.attack(self.features,self.adj, self.labels,self.idx_train, target_node, n_perturbations)  # FGA
            elif self.Attack_model == RND:
                model.attack(self.adj, self.labels, self.idx_train, target_node, n_perturbations)
            else:
                model.attack(self.features, self.adj, self.labels, target_node, n_perturbations, direct=True)
            modified_adj = model.modified_adj
            modified_features = model.modified_features
            # self.get_final_adj(modified_adj)

            if self.Attack_model in [RND,FGA,Nettack,IGAttack]:
                acc = self.single_test(modified_adj.toarray(), self.features.toarray(), target_node)
            else:
                # self.features = model.modified_features
                acc = self.single_test(modified_adj.toarray(), modified_features, target_node)
            # attacked_assortativity = self.compute_assortativity(torch.tensor(self.final_adj.nonzero()).long(), self.adj.shape[0])
            # original_assortativity = self.compute_assortativity(self.clean_data.edge_index, self.adj.shape[0])
            # similarity_score.append(1 - abs(attacked_assortativity - original_assortativity) / max(attacked_assortativity, original_assortativity))
            if acc == 0:
                cnt += 1
        print('Accuracy : %s' % (1 - cnt / num))
        # after_poison_overall_accuracy = self.test()

        after_poison_accuracy =1 - cnt / num
        return f"{after_poison_accuracy:.4f}"
                # f"{after_poison_overall_accuracy:.4f}",

    def compute_assortativity(self, edge_index, num_nodes):
        deg = degree(edge_index[0], num_nodes)
        # 构建度矩阵
        deg_i = deg[edge_index[0]]
        deg_j = deg[edge_index[1]]
        # 计算度同配性
        m = torch.mean(deg_i * deg_j)
        d = torch.mean(deg)
        v = torch.var(deg)
        # Pearson correlation coefficient
        assortativity = (m - d ** 2) / v
        return assortativity.item()

    def single_test(self,adj, features, target, gnn=None):
        copy_data = copy.deepcopy(self.pyg_data)
        copy_data.x = torch.tensor(features).to(device)
        copy_data.edge_index = torch.tensor(adj.nonzero()).to(device)

        if gnn is None:
            # test on GNN_model (poisoning attack)

            gnn = self.GNN_model(nfeat=self.nfeat,
                                  nhid=16,
                                  nclass=self.nclass,
                                  dropout=0.5,
                                 # args=args,
                                 device=device)

            gnn = gnn.to(device)
            gnn.fit(copy_data)
            # gnn.test()
            output = gnn.predict()
        else:
            # test on GNN_model (evasion attack)
            output = gnn.predict(x=copy_data.x, edge_index=copy_data.edge_index)
        probs = torch.exp(output[[target]])

        # acc_test = accuracy(output[[target]], labels[target])
        acc_test = (output.argmax(1)[target] == self.labels[target])
        del copy_data
        return acc_test.item()


    def multi_test_evasion(self):

        target_GNN_model = self.GNN_model(nfeat=self.nfeat,
                                     nhid=16,
                                     nclass=self.nclass,
                                     dropout=0.5,
                                          # args=args,
                                          device=device)

        target_GNN_model = target_GNN_model.to(device)


        target_GNN_model.fit(self.pyg_data)

        cnt = 0
        degrees = self.adj.sum(0).A1
        node_list = self.select_nodes(target_GNN_model)
        num = len(node_list)
        similarity_score=[]
        print('=== [Evasion] Attacking %s nodes respectively ===' % num)
        for target_node in tqdm(node_list):
            if isinstance(target_node, torch.Tensor):
                target_node = target_node.item()
            n_perturbations = int(degrees[target_node])

            if self.Attack_model == Ours:
                model = self.Attack_model(self.adj, self.surrogate, attack_structure=True, attack_features=False, device=device)
            elif self.Attack_model in [Nettack,IGAttack]:
                model = self.Attack_model(self.surrogate, nnodes=self.adj.shape[0], attack_structure=True, attack_features=False,
                                     device=device)
            elif self.Attack_model == FGA:
                model = FGA(self.surrogate, nnodes=self.adj.shape[0], device=device)
            elif self.Attack_model == RND:
                model = RND()
            elif self.Attack_model == SGAttack:
                model = self.Attack_model(self.surrogate, attack_structure=True, attack_features=False, device=device)

            model = model.to(device)
            if self.Attack_model in [FGA, IGAttack]:
                model.attack(self.features,self.adj, self.labels,self.idx_train, target_node, n_perturbations)  # FGA
                # self.features = model.modified_features
            elif self.Attack_model == RND:
                model.attack(self.adj, self.labels, self.idx_train, target_node, n_perturbations)
            else:
                model.attack(self.features, self.adj, self.labels, target_node, n_perturbations, direct=True)
                # self.features = model.modified_features
            modified_adj = model.modified_adj
            modified_features = model.modified_features
            # self.get_final_adj(modified_adj)


            if self.Attack_model in [FGA, RND,Nettack,IGAttack]:
                acc = self.single_test(modified_adj.toarray(), self.features.toarray(), target_node,gnn=target_GNN_model)
            else:
                acc = self.single_test(modified_adj.toarray(), modified_features, target_node,gnn=target_GNN_model)

            # attacked_assortativity = self.compute_assortativity(torch.tensor(self.final_adj.nonzero()).long(), self.adj.shape[0])
            # original_assortativity = self.compute_assortativity(self.clean_data.edge_index, self.adj.shape[0])
            # similarity_score.append(1 - abs(attacked_assortativity - original_assortativity) / max(attacked_assortativity, original_assortativity))
            if acc == 0:
                cnt += 1
        similarity_score = 1
        print('Accuracy : %s' % (1 - cnt / num))
        after_evasion_accuracy = 1 - cnt / num
        # after_evasion_overall_accuracy = self.test()
        # print('after_evasion_overall_accuracy' % after_evasion_overall_accuracy)


        return f"{after_evasion_accuracy:.4f}"
                # f"{after_evasion_overall_accuracy:.4f}" ,


