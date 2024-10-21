import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from collections import namedtuple
from functools import lru_cache

from scipy.sparse import csr_matrix
from torch_scatter import scatter_add
from torch_geometric.utils import k_hop_subgraph
from deeprobust.graph.targeted_attack import BaseAttack
from deeprobust.graph import utils
import scipy.sparse as sprs

SubGraph = namedtuple('SubGraph', ['edge_index', 'non_edge_index',
                                   'self_loop', 'self_loop_weight',
                                   'edge_weight', 'non_edge_weight',
                                   'edges_all'])


class Ours(BaseAttack):
    """SGAttack proposed in `Adversarial Attack on Large Scale Graph` TKDE 2021
    <https://arxiv.org/abs/2009.03488>

    SGAttack follows these steps::
    + training a surrogate SGC model with hop K
    + extrack a K-hop subgraph centered at target node
    + choose top-N attacker nodes that belong to the best wrong classes of the target node
    + compute gradients w.r.t to the subgraph to add or remove edges iteratively

    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    Examples
    --------

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import SGC
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> surrogate = SGC(nfeat=features.shape[1], K=3, lr=0.1,
              nclass=labels.max().item() + 1, device='cuda')
    >>> surrogate = surrogate.to('cuda')
    >>> pyg_data = Dpr2Pyg(data) # convert deeprobust dataset to pyg dataset
    >>> surrogate.fit(pyg_data, train_iters=200, patience=200, verbose=True) # train with earlystopping
    >>> from deeprobust.graph.targeted_attack import SGAttack
    >>> # Setup Attack Model
    >>> target_node = 0
    >>> model = SGAttack(surrogate, attack_structure=True, device=device)
    >>> # Attack
    >>> model.attack(features, adj, labels, target_node, n_perturbations=5)
    >>> modified_adj = model.modified_adj
    >>> modified_features = model.modified_features
    """

    def __init__(self,adj, model, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):

        super(Ours, self).__init__(model=None, nnodes=nnodes,
                                       attack_structure=attack_structure, attack_features=attack_features, device=device)

        self.target_node = None
        self.logits = model.predict()
        self.K = 2

        W = model.conv1.lin.weight.to(device)
        b = model.conv1.lin.bias
        if b is not None:
            b = b.to(device)

        self.weight, self.bias = W, b

        self.adjacency_matrix=adj

    @lru_cache(maxsize=1)
    def compute_XW(self):
        return F.linear(self.modified_features, self.weight)

    def get_ppr_score(self,adj, target, p=0.85, reversed=False):
        A = adj
        if reversed:
            A = A.T
        n, _ = A.shape
        r = np.asarray(A.sum(axis=1)).reshape(-1)
        k = r.nonzero()[0]
        D_1 = sprs.csr_matrix((1 / r[k], (k, k)), shape=(n, n))
        personalize = np.zeros(n)
        personalize[target] = 1
        personalize = personalize.reshape(n, 1)
        s = (personalize / personalize.sum()) * n
        I = sprs.eye(n)
        x = sprs.linalg.spsolve((I - p * A.T @ D_1), s)

        x = x / x.sum()
        return x

    def extract_subgraph_with_ppr_score(self,adj, ppr_score, top_n):
        adj_matrix = adj.tocsr(copy=False)
        col_indices = adj_matrix.indices
        row_offsets = adj_matrix.indptr
        nodes = np.argsort(ppr_score)[-top_n:]
        subgraph_nodes = set(nodes)
        subgraph_edges = []
        node_mapping = {node: index for index, node in enumerate(nodes)}
        for node in nodes:
            start = row_offsets[node]
            end = row_offsets[node + 1]

            for neighbor in col_indices[start:end]:
                if neighbor in subgraph_nodes:
                    subgraph_edges.append((node_mapping[node], node_mapping[neighbor]))

        edge_index = list(zip(*subgraph_edges))
        return (np.asarray(nodes), np.asarray(edge_index))

    def get_cosine_score(self,adj, target):
        adjacency_matrix = adj.copy()
        row_sums = np.array(adjacency_matrix.sum(1))[:, 0]
        row_indices, col_indices = adjacency_matrix.nonzero()
        adjacency_matrix.data = adjacency_matrix.data.astype(np.float64)
        adjacency_matrix.data /= row_sums[row_indices]
        node_i_vector = adjacency_matrix[target, :]
        cosine_similarities = node_i_vector * adjacency_matrix.T
        cosine_similarities_dense = cosine_similarities.toarray().flatten()
        return cosine_similarities_dense

    def extract_subgraph_with_cosine_score(self,adj, cosine_score, top_n):
        adj_matrix = adj.tocsr(copy=False)
        col_indices = adj_matrix.indices
        row_offsets = adj_matrix.indptr
        nodes = np.argsort(cosine_score)[-top_n:]
        subgraph_nodes = set(nodes)
        subgraph_edges = []
        node_mapping = {node: index for index, node in enumerate(nodes)}
        for node in nodes:
            start = row_offsets[node]
            end = row_offsets[node + 1]

            for neighbor in col_indices[start:end]:
                if neighbor in subgraph_nodes:
                    subgraph_edges.append((node_mapping[node], node_mapping[neighbor]))

        edge_index = list(zip(*subgraph_edges))
        return (np.asarray(nodes), np.asarray(edge_index))

    def attack(self, features, adj, labels, target_node, n_perturbations, direct=True, n_influencers=3, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        features :
            Original (unperturbed) node feature matrix
        adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        target_node : int
            target_node node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        direct: bool
            whether to conduct direct attack
        n_influencers : int
            number of the top influencers to choose. For direct attack, it will set as `n_perturbations`.
        """
        if sp.issparse(features):
            # to dense numpy matrix
            features = features.A

        if not torch.is_tensor(features):
            features = torch.tensor(features, device=self.device)

        if torch.is_tensor(adj):
            adj = utils.to_scipy(adj).csr()

        self.modified_features = features.requires_grad_(bool(self.attack_features))

        target_label = torch.LongTensor([labels[target_node]])
        best_wrong_label = torch.LongTensor([(self.logits[target_node].cpu() - 1000 * torch.eye(self.logits.size(1))[target_label]).argmax()])

        self.selfloop_degree = torch.tensor(adj.sum(1).A1 + 1, device=self.device)
        self.target_label = target_label.to(self.device)
        self.best_wrong_label = best_wrong_label.to(self.device)
        self.n_perturbations = n_perturbations
        self.ori_adj = adj
        self.target_node = target_node
        self.direct = direct

        attacker_nodes = torch.where(torch.as_tensor(labels) == best_wrong_label)[0]
        subgraph = self.get_subgraph(attacker_nodes, n_influencers)

        if not direct:
            # for indirect attack, the edges adjacent to targeted node should not be considered
            mask = torch.logical_or(subgraph.edge_index[0] == target_node, subgraph.edge_index[1] == target_node).to(self.device)

        structure_perturbations = []
        feature_perturbations = []
        num_features = features.shape[-1]
        for _ in range(n_perturbations):
            edge_grad, non_edge_grad, features_grad = self.compute_gradient(subgraph)
            max_structure_score = max_feature_score = 0.

            if self.attack_structure:
                edge_grad *= (-2 * subgraph.edge_weight + 1)
                non_edge_grad *= -2 * subgraph.non_edge_weight + 1
                min_grad = min(edge_grad.min().item(), non_edge_grad.min().item())
                edge_grad -= min_grad
                non_edge_grad -= min_grad
                if not direct:
                    edge_grad[mask] = 0.
                max_edge_grad, max_edge_idx = torch.max(edge_grad, dim=0)
                max_non_edge_grad, max_non_edge_idx = torch.max(non_edge_grad, dim=0)
                max_structure_score = max(max_edge_grad.item(), max_non_edge_grad.item())

            if self.attack_features:
                features_grad *= -2 * self.modified_features + 1
                features_grad -= features_grad.min()
                if not direct:
                    features_grad[target_node] = 0.
                max_feature_grad, max_feature_idx = torch.max(features_grad.view(-1), dim=0)
                max_feature_score = max_feature_grad.item()

            if max_structure_score >= max_feature_score:
                if max_edge_grad > max_non_edge_grad:
                    # remove one edge
                    best_edge = subgraph.edge_index[:, max_edge_idx]
                    subgraph.edge_weight.data[max_edge_idx] = 0
                    self.selfloop_degree[best_edge] -= 1
                else:
                    # add one edge
                    best_edge = subgraph.non_edge_index[:, max_non_edge_idx]
                    subgraph.non_edge_weight.data[max_non_edge_idx] = 1
                    self.selfloop_degree[best_edge] += 1

                u, v = best_edge.tolist()
                structure_perturbations.append((u, v))
            else:
                u, v = divmod(max_feature_idx.item(), num_features)
                feature_perturbations.append((u, v))
                self.modified_features[u, v].data.fill_(1. - self.modified_features[u, v].data)

        if structure_perturbations:
            modified_adj = adj.tolil(copy=True)
            row, col = list(zip(*structure_perturbations))
            modified_adj[row, col] = modified_adj[col, row] = 1 - modified_adj[row, col].A
            modified_adj = modified_adj.tocsr(copy=False)
            modified_adj.eliminate_zeros()
        else:
            modified_adj = adj.copy()

        self.modified_adj = modified_adj
        self.modified_features = self.modified_features.detach().cpu().numpy()
        self.structure_perturbations = structure_perturbations
        self.feature_perturbations = feature_perturbations

    def get_subgraph(self, attacker_nodes, n_influencers=None):
        target_node = self.target_node
        neighbors = self.ori_adj[target_node].indices

        #整张图提取子图：时间太长
        # ppr_score = self.get_ppr_score(self.adjacency_matrix, target_node, 0.85)
        # self.sub_ppr_nodes, self.sub_ppr_edges = self.extract_subgraph_with_ppr_score(self.adjacency_matrix, ppr_score, 200)
        # cos_score = self.get_cosine_score(self.adjacency_matrix, target_node)
        # self.sub_cos_nodes, self.sub_cos_edges = self.extract_subgraph_with_cosine_score(self.adjacency_matrix, cos_score, 200)

        sub_k_nodes, sub_k_edges = self.ego_subgraph()
        sub_k_nodes = sub_k_nodes.cpu().numpy()
        sub_k_edges = sub_k_edges.cpu().numpy()

        #提取k跳子图的子图
        adj_k_matrix = torch.zeros((self.adjacency_matrix.toarray()[1].size, self.adjacency_matrix.toarray()[1].size), dtype=torch.float32)
        for i, j in zip(sub_k_edges[0], sub_k_edges[1]):
            adj_k_matrix[i, j] = 1
            adj_k_matrix[j, i] = 1  # 邻接矩阵对称
        adj_k_matrix = csr_matrix(adj_k_matrix)
        ppr_score = self.get_ppr_score(adj_k_matrix, target_node, 0.85)
        self.sub_ppr_nodes, self.sub_ppr_edges = self.extract_subgraph_with_ppr_score(adj_k_matrix, ppr_score, 150)
        cos_score = self.get_cosine_score(adj_k_matrix, target_node)
        self.sub_cos_nodes, self.sub_cos_edges = self.extract_subgraph_with_cosine_score(adj_k_matrix, cos_score, 150)

        #综合
        overlap = np.union1d(self.sub_ppr_nodes, self.sub_cos_nodes)
        overlap = np.intersect1d(overlap, sub_k_nodes)
        if len(sub_k_edges) == 2:
            U, V = sub_k_edges
        else:
            print(f"Unexpected structure or empty sub_edges3: {sub_k_edges}")
            U, V = [], []  # or handle this case appropriately
        final_edge = [[], []]
        for i in range(len(U)):
            u, v = U[i], V[i]
            if u in overlap and v in overlap:
                final_edge[0].append(u)
                final_edge[1].append(v)

        sub_nodes, sub_edges = np.asarray(overlap), np.asarray(final_edge)
        # 获取子图节点的ppr分数
        sub_nodes_ppr_scores = ppr_score[sub_nodes]
        # 获取子图节点的cos分数
        sub_nodes_cos_scores = cos_score[sub_nodes]

        sub_nodes = torch.from_numpy(sub_nodes).to(self.device)
        sub_edges = torch.from_numpy(sub_edges).to(self.device)

        self.target_edges = torch.stack((sub_nodes, torch.full(sub_nodes.size(), self.target_node, device=self.device)),dim=0)


        a=0.7
        self.target_edge_weight= torch.from_numpy(a * sub_nodes_ppr_scores + (1-a) * sub_nodes_cos_scores).to(self.device).requires_grad_()

        
        if self.direct or n_influencers is not None:
            influencers = [target_node]
            attacker_nodes = np.setdiff1d(attacker_nodes, neighbors)
        else:
            influencers = neighbors

        subgraph = self.subgraph_processing(influencers, attacker_nodes, sub_nodes, sub_edges)

        if n_influencers is not None and self.attack_structure:
            if self.direct:
                influencers = [target_node]
                attacker_nodes = self.get_topk_influencers(subgraph, k=self.n_perturbations + 1)

            else:
                influencers = neighbors
                attacker_nodes = self.get_topk_influencers(subgraph, k=n_influencers)

            subgraph = self.subgraph_processing(influencers, attacker_nodes, sub_nodes, sub_edges)
        return subgraph

    def get_topk_influencers(self, subgraph, k):
        _, non_edge_grad, _ = self.compute_gradient(subgraph)
        _, topk_nodes = torch.topk(non_edge_grad, k=k, sorted=False)

        influencers = subgraph.non_edge_index[1][topk_nodes.cpu()]
        return influencers.cpu().numpy()

    def subgraph_processing(self, influencers, attacker_nodes, sub_nodes, sub_edges):
        if not self.attack_structure:
            self_loop = sub_nodes.repeat((2, 1))
            edges_all = torch.cat([sub_edges, sub_edges[[1, 0]], self_loop], dim=1)
            edge_weight = torch.ones(edges_all.size(1), device=self.device)

            return SubGraph(edge_index=sub_edges, non_edge_index=None,
                            self_loop=None, edges_all=edges_all,
                            edge_weight=edge_weight, non_edge_weight=None,
                            self_loop_weight=None)

        row = np.repeat(influencers, len(attacker_nodes))
        col = np.tile(attacker_nodes, len(influencers))
        non_edges = np.row_stack([row, col])

        if len(influencers) > 1:
            mask = self.ori_adj[non_edges[0],
                                non_edges[1]].A1 == 0
            non_edges = non_edges[:, mask]

        non_edges = torch.as_tensor(non_edges, device=self.device)
        unique_nodes = np.union1d(sub_nodes.tolist(), attacker_nodes)
        unique_nodes = torch.as_tensor(unique_nodes, device=self.device)
        self_loop = unique_nodes.repeat((2, 1))
        edges_all = torch.cat([sub_edges, sub_edges[[1, 0]],
                               non_edges, non_edges[[1, 0]], self_loop, self.target_edges], dim=1)

        edge_weight = torch.ones(sub_edges.size(1), device=self.device).requires_grad_(bool(self.attack_structure))
        non_edge_weight = torch.zeros(non_edges.size(1), device=self.device).requires_grad_(bool(self.attack_structure))
        self_loop_weight = torch.ones(self_loop.size(1), device=self.device)

        edge_index = sub_edges
        non_edge_index = non_edges
        self_loop = self_loop

        subgraph = SubGraph(edge_index=edge_index, non_edge_index=non_edge_index,
                            self_loop=self_loop, edges_all=edges_all,
                            edge_weight=edge_weight, non_edge_weight=non_edge_weight,
                            self_loop_weight=self_loop_weight)
        return subgraph

    def SGCCov(self, x, edge_index, edge_weight):
        row, col = edge_index
        for _ in range(self.K):
            src = x[row] * edge_weight.view(-1, 1)
            x = scatter_add(src, col, dim=-2, dim_size=x.size(0))
        return x

    def compute_gradient(self, subgraph, eps=5.0):
        if self.attack_structure:
            edge_weight = subgraph.edge_weight
            non_edge_weight = subgraph.non_edge_weight
            self_loop_weight = subgraph.self_loop_weight
            weights = torch.cat([edge_weight, edge_weight,
                                non_edge_weight, non_edge_weight,
                                self_loop_weight, self.target_edge_weight], dim=0)
        else:
            weights = subgraph.edge_weight

        weights = self.gcn_norm(subgraph.edges_all, weights, self.selfloop_degree)
        logit = self.SGCCov(self.compute_XW(), subgraph.edges_all, weights)
        logit = logit[self.target_node]
        if self.bias is not None:
            logit += self.bias

        # model calibration
        logit = F.log_softmax(logit.view(1, -1) / eps, dim=1)
        loss = F.nll_loss(logit, self.target_label) - F.nll_loss(logit, self.best_wrong_label)

        edge_grad = non_edge_grad = features_grad = None

        if self.attack_structure and self.attack_features:
            edge_grad, non_edge_grad, features_grad = torch.autograd.grad(loss, [edge_weight, non_edge_weight, self.modified_features], create_graph=False)

        elif self.attack_structure:
            edge_grad, non_edge_grad = torch.autograd.grad(loss, [edge_weight, non_edge_weight], create_graph=False)
        else:
            features_grad = torch.autograd.grad(loss, self.modified_features, create_graph=False)[0]

        if self.attack_features:
            self.compute_XW.cache_clear()
        return edge_grad, non_edge_grad, features_grad

    def ego_subgraph(self):
        edge_index = np.asarray(self.ori_adj.nonzero())
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=self.device)
        sub_nodes, sub_edges, *_ = k_hop_subgraph(int(self.target_node), self.K, edge_index)
        sub_edges = sub_edges[:, sub_edges[0] < sub_edges[1]]

        return sub_nodes, sub_edges

    @ staticmethod
    def gcn_norm(edge_index, weights, degree):
        row, col = edge_index
        inv_degree = torch.pow(degree, -0.5)
        normed_weights = weights * inv_degree[row] * inv_degree[col]
        return normed_weights
