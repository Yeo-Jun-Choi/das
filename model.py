import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self, args, user_degree, item_degree):
        super(model, self).__init__()
        self.user_num = args.user_num
        self.item_num = args.item_num
        self.embedding_dim = args.embedding_dim

        self.negative_num = args.negative_num
        self.gamma = args.gamma
        
        self.user_degree = user_degree.to(args.device)
        self.user_max_degree = args.user_max_degree
        self.item_degree = item_degree.to(args.device)
        self.item_max_degree = args.item_max_degree
        self.weight = args.weight
        self.margin1 = args.margin1
        self.margin2 = args.margin2
        
        self.user_embeds = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim)

        self.initial_weight = args.initial_weight
        self.initial_weights()

    def initial_weights(self):
        nn.init.normal_(self.user_embeds.weight, std=self.initial_weight)
        nn.init.normal_(self.item_embeds.weight, std=self.initial_weight)

    def cal_loss_L(self, users, pos_items, neg_items):
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)

        user_norm = F.normalize(user_embeds, dim=-1)
        pos_norm = F.normalize(pos_embeds, dim=-1)
        neg_norm = F.normalize(neg_embeds, dim=-1)
        user_pos_weight = self.user_pop_weight(users)
        pos_weight = self.item_pop_weight(pos_items)
        norm_pos_weight = self.item_pop_weight_norm(pos_items)
        norm_user_weight = self.user_pop_weight_norm(users)
        

        pos_scores = (user_norm * pos_embeds).sum(dim=-1)
        neg_scores = (user_norm.unsqueeze(1) * neg_embeds).sum(dim=-1) 
        norm_pos_scores = (user_norm * pos_norm).sum(dim=-1)
        norm_neg_scores = (user_norm.unsqueeze(1) * neg_norm).sum(dim=-1) 

        loss1 = self.cal_softmax_inner(pos_scores, neg_scores, pos_weight, user_pos_weight)
        loss2 = self.cal_softmax_cosine(norm_pos_scores, norm_neg_scores, norm_pos_weight, norm_user_weight)
        
        loss = loss1 * self.weight + loss2

        return loss, self.weight * loss1, loss2
    
    def cal_softmax_inner(self, pos, neg, pos_weight, user_pos_weight):
        exp_pos = torch.exp((pos + pos_weight + user_pos_weight)/self.margin1)
        exp_neg = torch.exp(neg/self.margin1)
        neg_sum = torch.mean(exp_neg, dim=-1)
        
        denominator = (self.negative_num*neg_sum) + exp_pos + 1e-7
        numerator = exp_pos
        loss = -torch.log(numerator/denominator).mean()
        return loss

    def cal_softmax_cosine(self, pos_norm, neg_norm, pos_weight, user_pos_weight):
        exp_pos = torch.exp((pos_norm + pos_weight + user_pos_weight)/self.margin2)
        exp_neg = torch.exp(neg_norm/self.margin2)
        neg_sum = torch.mean(exp_neg, dim=-1)

        denominator = (self.negative_num*neg_sum) + exp_pos + 1e-7
        numerator = exp_pos
        loss = -torch.log(numerator/denominator).mean()
        return loss

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def forward(self, users, pos_items, neg_items):
        loss, pos, neg = self.cal_loss_L(users, pos_items, neg_items)
        loss += self.gamma * self.norm_loss()
        return loss, pos, neg, self.gamma * self.norm_loss()

    def test_foward(self, users):
        items = torch.arange(self.item_num).to(users.device)
        user_embeds = self.user_embeds(users)
        item_embeds = self.item_embeds(items)
        return user_embeds.mm(item_embeds.t())

    def get_device(self):
        return self.user_embeds.weight.device

    def user_pop_weight(self, user_index=None):
        if user_index == None:
            degree = self.user_degree
        else:
            degree = self.user_degree[user_index]
        pop = torch.log(degree * self.user_max_degree)
        return pop

    def item_pop_weight(self, item_index=None):
        if item_index == None:
            degree = self.item_degree
        else:
            degree = self.item_degree[item_index]
        pop = torch.log(degree * self.item_max_degree)
        return pop

    def item_pop_weight_norm(self, item_index=None):
        if item_index == None:
            degree = self.item_degree
        else:
            degree = self.item_degree[item_index]
        pop = torch.log(degree * self.item_max_degree)
        pop = pop / (-torch.log(torch.min(self.item_degree) * self.item_max_degree + 1e-7))
        return pop
    
    def user_pop_weight_norm(self, user_index=None):
        if user_index == None:
            degree = self.user_degree
        else:
            degree = self.user_degree[user_index]
        pop = torch.log(degree * self.user_max_degree)
        pop = pop / (-torch.log(torch.min(self.user_degree) * self.user_max_degree + 1e-7))
        return pop