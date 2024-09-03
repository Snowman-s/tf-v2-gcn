import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

def encode_onehot(labels):
    classes = list(set(labels))
    class_to_index = {c: i for i, c in enumerate(classes)}
    labels_onehot = np.eye(len(classes), dtype=np.int32)[[class_to_index[label] for label in labels]]
    return labels_onehot

def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print(f'Loading {dataset} dataset...')

    idx_features_labels = np.genfromtxt(f"{path}{dataset}.content", dtype=str)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(f"{path}{dataset}.cites", dtype=np.int32)
    edges = np.array([idx_map.get(edge) for edge in edges_unordered.flatten()]).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print(f'Dataset has {adj.shape[0]} nodes, {edges.shape[0]} edges, {features.shape[1]} features.')

    return features.todense(), adj, labels


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    return normalize_adj(adj, symmetric)

def sample_mask(idx, length):
    mask = np.zeros(length, dtype=bool)
    mask[idx] = True
    return mask

def get_splits(y):
    idx_train = slice(140)
    idx_val = slice(200, 500)
    idx_test = slice(500, 1500)
    y_train, y_val, y_test = np.zeros_like(y, dtype=np.int32), np.zeros_like(y, dtype=np.int32), np.zeros_like(y, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    val_mask = sample_mask(idx_val, y.shape[0])
    test_mask = sample_mask(idx_test, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask, val_mask, test_mask

def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.sum(labels * preds, axis=1)))

def accuracy(preds, labels):
    return np.mean(np.argmax(labels, axis=1) == np.argmax(preds, axis=1))

#def evaluate_preds(preds, labels, indices):
#    split_loss = []
#    split_acc = []
#
#    for y_split, idx_split in zip(labels, indices):
#        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
#        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))
#
#    return split_loss, split_acc

def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    return sp.eye(adj.shape[0]) - adj_normalized

def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    return (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])

def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print(f"Calculating Chebyshev polynomials up to order {k}...")

    T_k = [sp.eye(X.shape[0], format='csr'), X]

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = X.copy()
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for _ in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).T
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
