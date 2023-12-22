"""
(X. He et al., Neural collaborative filtering, WWW17')

GMF, NCF(MLP compoment), NeuMF
"""
import torch
import torch.nn as nn

class Generalized_Matrix_Factorization(nn.Module):
    """
    element-wise dot product를 통한 matrix factorization.
    user와 item 간 상호작용의 선형적인 관계를 포착
    """
    def __init__(self, args, num_users, num_items):
        super(Generalized_Matrix_Factorization, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items

        self.embed_dim = args.embed_dim

        # Define user/item representation vector
        self.embed_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embed_dim)
        self.embed_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embed_dim)

        # (u, v) -> r
        self.output = nn.Linear(in_features=self.embed_dim, out_features=1)

    def forward(self, user_indices, item_indices):
        user_embedding_vector = self.embed_user(user_indices)
        item_embedding_vector = self.embed_item(item_indices)

        # Perform matrix factorization(element-wise dot product), using user/item embedding vector
        factorized_vector = torch.mul(user_embedding_vector, item_embedding_vector)

        predicted_rating = self.output(factorized_vector)

        return predicted_rating.squeeze()

class Neural_Collaborative_Filtering(nn.Module):
    """
    Multi-layer perceptron (MLP)을 이용한 collaborative filtering.
    user와 item 간 상호작용의 비선형적인 관계를 포착
    """
    def __init__(self, args, num_users, num_items):
        super(Neural_Collaborative_Filtering, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = int(int(args.layers[0]) / 2)
        self.layers = args.layers
        self.dropout = args.dropout

        # Define user/item representation vector (for input)
        self.embed_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embed_dim)
        self.embed_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embed_dim)

        # Stack MLP layers with Batch Normalization
        self.mlp_layers = nn.ModuleList()
        for _, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.mlp_layers.append(nn.Linear(input_size, output_size))
            self.mlp_layers.append(nn.BatchNorm1d(output_size))
            self.mlp_layers.append(nn.Dropout(p=args.dropout))
            self.mlp_layers.append(nn.ReLU())

        # (u, v) -> r        
        self.output = nn.Linear(in_features=self.layers[-1], out_features=1)

    def forward(self, user_indices, item_indices):
        user_embedding_vector = self.embed_user(user_indices)
        item_embedding_vector = self.embed_item(item_indices)

        # concat user/item vector
        vector = torch.cat([user_embedding_vector, item_embedding_vector], dim=-1)

        # fed vector into MLP layers
        for idx, _ in enumerate(range(len(self.mlp_layers))):
            vector = self.mlp_layers[idx](vector)

        predicted_rating = self.output(vector)

        return predicted_rating.squeeze()

class Neural_Matrix_Factorization(nn.Module):
    """
    GMF + NCF = NeuMF
    """
    def __init__(self, args, num_users, num_items):
        super(Neural_Matrix_Factorization, self).__init__()

        self.num_users = num_users
        self.num_items = num_items

        self.embed_dim_mf = args.embed_dim
        self.embed_dim_mlp = int(int(args.layers[0]) / 2)

        self.mlp_layers = args.layers
        self.dropout = args.dropout

        # GMF component
        self.embed_user_mf = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embed_dim_mf)
        self.embed_item_mf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embed_dim_mf)

        # MLP component
        self.embed_user_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embed_dim_mlp)
        self.embed_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embed_dim_mlp)

        # Stack MLP layers with Batch Normalization
        self.mlp_layers = nn.ModuleList()
        for _, (input_size, output_size) in enumerate(zip(args.layers[:-1], args.layers[1:])):
            self.mlp_layers.append(nn.Linear(input_size, output_size))
            self.mlp_layers.append(nn.BatchNorm1d(output_size))
            self.mlp_layers.append(nn.Dropout(p=args.dropout))
            self.mlp_layers.append(nn.ReLU())

        self.final_output = nn.Linear(in_features=args.layers[-1] + self.embed_dim_mf, out_features=1)
    
    def forward(self, user_indices, item_indices):
        user_embed_mf = self.embed_user_mf(user_indices)
        item_embed_mf = self.embed_item_mf(item_indices)
        
        # element-wise dot product
        mf_vector = torch.mul(user_embed_mf, item_embed_mf)

        user_embed_mlp = self.embed_user_mlp(user_indices)
        item_embed_mlp = self.embed_item_mlp(item_indices)

        # concat 2 vector for MLP input & fed into MLP layers
        mlp_vector = torch.cat([user_embed_mlp, item_embed_mlp], dim=-1)
        for idx, _ in enumerate(range(len(self.mlp_layers))):
            mlp_vector = self.mlp_layers[idx](mlp_vector)
        
        # concat MF vector & MLP vector
        final_vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        rating = self.final_output(final_vector)

        return rating.squeeze()
        