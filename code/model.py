import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import MGCATConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention(nn.Module):
    def __init__(self, in_size):
        super(Attention, self).__init__()
        hidden_size = in_size

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class Model(nn.Module):
    def __init__(self, dataset, NUM_LAYERS, EMBEDDING_SIZE, HEADS):
        super(Model, self).__init__()
        self.NUM_LAYERS = NUM_LAYERS
        self.edge_index = dataset.edge_index.to(device)

        self.lncRNA_feature = []
        self.miRNA_feature = []
        self.lncRNA_feature_proj = []
        self.miRNA_feature_proj = []
        lncRNA_feature = dataset.lncRNA_feature
        miRNA_feature = dataset.miRNA_feature
        for lf, mf in zip(lncRNA_feature, miRNA_feature):
            self.lncRNA_feature.append(torch.tensor(lf, dtype=torch.float32).to(device))
            self.miRNA_feature.append(torch.tensor(mf, dtype=torch.float32).to(device))
            self.lncRNA_feature_proj.append(nn.Linear(lf.shape[1], EMBEDDING_SIZE).to(device))
            self.miRNA_feature_proj.append(nn.Linear(mf.shape[1], EMBEDDING_SIZE).to(device))

        self.conv1 = MGCATConv(EMBEDDING_SIZE, EMBEDDING_SIZE, heads=HEADS)
        self.norm1 = nn.BatchNorm1d(EMBEDDING_SIZE)
        self.conv2 = MGCATConv(EMBEDDING_SIZE, EMBEDDING_SIZE, heads=HEADS)
        self.norm2 = nn.BatchNorm1d(EMBEDDING_SIZE)
        self.conv3 = MGCATConv(EMBEDDING_SIZE, EMBEDDING_SIZE, heads=HEADS)
        self.norm3 = nn.BatchNorm1d(EMBEDDING_SIZE)
        self.conv4 = MGCATConv(EMBEDDING_SIZE, EMBEDDING_SIZE, heads=HEADS)

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(EMBEDDING_SIZE * 2),
            nn.Linear(EMBEDDING_SIZE * 2, EMBEDDING_SIZE),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(EMBEDDING_SIZE),
            nn.Linear(EMBEDDING_SIZE, 1),
            nn.Sigmoid(),
        )

        self.view_att = Attention(EMBEDDING_SIZE)
        self.layer_att = Attention(EMBEDDING_SIZE)

        self.criterion = torch.nn.BCELoss(reduction='sum')

    def project(self):
        projected_lncRNA_features = []
        projected_miRNA_features = []
        for idx, (lf, mf) in enumerate(zip(self.lncRNA_feature, self.miRNA_feature)):
            projected_lncRNA_feature = self.lncRNA_feature_proj[idx](lf)
            projected_lncRNA_features.append(projected_lncRNA_feature)
            projected_miRNA_feature = self.miRNA_feature_proj[idx](mf)
            projected_miRNA_features.append(projected_miRNA_feature)
        return projected_lncRNA_features, projected_miRNA_features

    def forward(self, idx, lbl=None):
        projected_lncRNA_features, projected_miRNA_features = self.project()
        lncRNA_features = torch.stack(projected_lncRNA_features, dim=1)
        miRNA_features = torch.stack(projected_miRNA_features, dim=1)
        view_features = torch.concat([lncRNA_features, miRNA_features], dim=0)
        view_features, _ = self.view_att(view_features)

        x0 = view_features
        x_list = [x0]
        x0 = F.dropout(x0, p=0.1)
        x1 = self.conv1(x0, self.edge_index)
        x_list.append(x1)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)
        x2 = self.conv2(x1, self.edge_index)
        x_list.append(x2)
        x2 = self.norm2(x2)
        x2 = F.relu(x2)
        x3 = self.conv3(x2, self.edge_index)
        x_list.append(x3)
        x3 = self.norm3(x3)
        x3 = F.relu(x3)
        x4 = self.conv4(x3, self.edge_index)
        x_list.append(x4)

        x_features = torch.stack(x_list, dim=1)
        x, _ = self.layer_att(x_features)

        u_feature = x[idx[0]]
        v_feature = x[idx[1]]
        uv_feature = torch.cat((u_feature, v_feature), dim=1)
        out = self.mlp(uv_feature)
        out = torch.squeeze(out)
        if lbl is None:
            return out, None
        loss_train = self.criterion(out, lbl.float())
        return out, loss_train
