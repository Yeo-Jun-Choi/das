
import torch
import torch.nn.functional as F
import pickle
import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
import os
import argparse
import random
from model import model


seed = 2020
random.seed(seed)
np.random.seed(seed)
os.environ['PL_GLOVAL_SEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# deterministic
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True


def data_param_prepare(args):
    train_data, test_data, train_mat, user_num, item_num = load_data(args.train_file_path, args.test_file_path)
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle = True)
    test_loader = data.DataLoader(list(range(user_num)), batch_size=args.test_batch_size, shuffle=False)

    args.user_num = user_num
    args.item_num = item_num
    
    mask = torch.zeros(user_num, item_num)
    interacted_items = [[] for _ in range(user_num)]
    for (u, i) in train_data:
        mask[u][i] = -np.inf
        interacted_items[u].append(i)

    test_ground_truth_list = [[] for _ in range(user_num)]
    for (u, i) in test_data:
        test_ground_truth_list[u].append(i)

    csr_matrix = train_mat.tocsc()
    user_degrees = np.diff(train_mat.tocsr().indptr)
    item_degrees = np.diff(csr_matrix.indptr)
    
    user_degrees = torch.from_numpy(user_degrees)
    item_degrees = torch.from_numpy(item_degrees)

    return args, train_loader, test_loader, train_mat, test_ground_truth_list, mask, user_degrees, item_degrees

    
def load_data(train_file, test_file):
    trainUniqueUsers, trainItem, trainUser = [], [], []
    testUniqueUsers, testItem, testUser = [], [], []
    n_user, m_item = 0, 0
    trainDataSize, testDataSize = 0, 0
    with open(train_file, 'r') as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                trainUniqueUsers.append(uid)
                trainUser.extend([uid] * len(items))
                trainItem.extend(items)
                m_item = max(m_item, max(items))
                n_user = max(n_user, uid)
                trainDataSize += len(items)
    trainUniqueUsers = np.array(trainUniqueUsers)
    trainUser = np.array(trainUser)
    trainItem = np.array(trainItem)

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                try:
                    items = [int(i) for i in l[1:]]
                except:
                    items = []
                uid = int(l[0])
                testUniqueUsers.append(uid)
                testUser.extend([uid] * len(items))
                testItem.extend(items)
                try:
                    m_item = max(m_item, max(items))
                except:
                    m_item = m_item
                n_user = max(n_user, uid)
                testDataSize += len(items)

    train_data = []
    test_data = []

    n_user += 1
    m_item += 1

    for i in range(len(trainUser)):
        train_data.append([trainUser[i], trainItem[i]])
    for i in range(len(testUser)):
        test_data.append([testUser[i], testItem[i]])
    train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0
    return train_data, test_data, train_mat, n_user, m_item


def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	return res

def pstore(x, path):
	with open(path, 'wb') as f:
		pickle.dump(x, f)

def Sampling(pos_train_data, neg_ratio, neg_candidates):
    batch_size = len(pos_train_data[0])
    neg_items = np.random.choice(neg_candidates, (batch_size, neg_ratio), replace = True)
    neg_items = torch.from_numpy(neg_items)
 
    return pos_train_data[0], pos_train_data[1], neg_items


########################### TRAINING #####################################

def train(model, optimizer, train_loader, test_loader, test_ground_truth_list, mask, train_mat, args): 
    device = args.device
    best_epoch, best_recall, best_ndcg = 0, 0, 0
    early_stop_count = 0
    early_stop = False

    batches = len(train_loader.dataset) // args.batch_size
    if len(train_loader.dataset) % args.batch_size != 0:
        batches += 1
    neg_candidates = np.arange(args.item_num)
    for epoch in range(args.max_epoch):
        model.train() 

        for batch, x in enumerate(train_loader):
            users, pos_items, neg_items = Sampling(x, args.negative_num, neg_candidates)
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            model.zero_grad()
            loss, pos, neg, regul = model(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()

        need_test = True
        if need_test:
            F1_score, Precision, Recall, NDCG = test(model, test_loader, test_ground_truth_list, mask, args.topk, args.user_num)
            print("Loss = {:.5f}, Pos = {:.5f}, Neg = {:.5f}, regul = {:.5f}, F1-score: {:5f}  Precision: {:.5f}  Recall: {:.5f} NDCG: {:.5f}".format(loss.item(), pos.item(), neg.item(), regul.item(), F1_score, Precision, Recall, NDCG))
            
            if Recall > best_recall:
                best_recall, best_ndcg, best_epoch = Recall, NDCG, epoch
                early_stop_count = 0
                torch.save(model.state_dict(), args.model_save_path)

            elif epoch < args.start_early:
                pass

            else:
                early_stop_count += 1
                if early_stop_count == args.early_stop_epoch:
                    early_stop = True
        
        if early_stop:
            print('Early stop is triggered at {} epochs.'.format(epoch))
            print('best epoch = {}, best recall = {}, best ndcg = {}'.format(best_epoch, best_recall, best_ndcg))
            print('The best model is saved at {}'.format(args.model_save_path))
            break

    print('Training end!')


########################### TESTING #####################################

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def RecallPrecision_ATk(test_data, r, k):
	right_pred = r[:, :k].sum(1)
	precis_n = k
	
	recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
	recall_n = np.where(recall_n != 0, recall_n, 1)
	recall = np.sum(right_pred / recall_n)
	precis = np.sum(right_pred) / precis_n
	return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
	pred_data = r[:, :k]
	scores = np.log2(1. / np.arange(1, k + 1))
	pred_data = pred_data / scores
	pred_data = pred_data.sum(1)
	return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
	assert len(r) == len(test_data)
	pred_data = r[:, :k]

	test_matrix = np.zeros((len(pred_data), k))
	for i, items in enumerate(test_data):
		length = k if k <= len(items) else len(items)
		test_matrix[i, :length] = 1
	max_r = test_matrix
	idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
	dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
	dcg = np.sum(dcg, axis=1)
	idcg[idcg == 0.] = 1.
	ndcg = dcg / idcg
	ndcg[np.isnan(ndcg)] = 0.
	return np.sum(ndcg)


def test_one_batch(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    ret = RecallPrecision_ATk(groundTrue, r, k)
    return ret['precision'], ret['recall'], NDCGatK_r(groundTrue,r,k)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def test(model, test_loader, test_ground_truth_list, mask, topk, n_user):
    rating_list = []
    groundTrue_list = []

    with torch.no_grad():
        model.eval()
        for idx, batch_users in enumerate(test_loader):
            
            batch_users = batch_users.to(model.get_device())
            rating = model.test_foward(batch_users) 
            rating = rating.cpu()
            rating += mask[batch_users.cpu()]
            
            _, rating_K = torch.topk(rating, k=topk)
            rating_list.append(rating_K)

            groundTrue_list.append([test_ground_truth_list[u] for u in batch_users])

    X = zip(rating_list, groundTrue_list)
    Recall, Precision, NDCG = 0, 0, 0

    for i, x in enumerate(X):
        precision, recall, ndcg = test_one_batch(x, topk)
        Recall += recall
        Precision += precision
        NDCG += ndcg
        
    Precision /= n_user
    Recall /= n_user
    NDCG /= n_user
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)

    return F1_score, Precision, Recall, NDCG


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    
    ## training ## 
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=2000)
    parser.add_argument('--initial_weight', type=float, default=1e-3)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--early_stop_epoch', type=int, default=20)
    parser.add_argument('--start_early', type=int, default=10)
    parser.add_argument('--negative_num', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=1e-7)

    ## test ##
    parser.add_argument('--test_batch_size', type=int, default=2048)
    parser.add_argument('--topk', type=int, default=20)

    ## dataset ##
    parser.add_argument('--dataset', type=str, default='Movielens1M_m1')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model_save_path', type=str, default='None')

    ## weights ##
    parser.add_argument('--margin1', type=float, default=1.0)
    parser.add_argument('--margin2', type=float, default=1.0)
    parser.add_argument('--weight', type=float, default=0.2)
    args = parser.parse_args()

    args.model_save_path = f'./{args.dataset}_{args.margin1}_{args.margin2}_{args.weight}_main.pt'
    args.train_file_path = f'./data/{args.dataset}/train.txt'
    args.test_file_path = f'./data/{args.dataset}/test.txt'

    args, train_loader, test_loader, train_mat, ground_true, mask, user_degree, item_degree = data_param_prepare(args)
    train_mat = train_mat.toarray()
    
    args.user_max_degree = 1/user_degree.max()
    args.item_max_degree = 1/item_degree.max()
    
    model = model(args, user_degree, item_degree)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train(model, optimizer, train_loader, test_loader, ground_true, mask, train_mat, args)

    exit()
