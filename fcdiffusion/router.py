import torch
import torch.nn as nn

class PromptComplexityRouter(nn.Module):
    """
    一个简单的MLP网络，用于根据文本特征预测视觉复杂度。
    这是一个回归模型，输出一个连续的复杂度分数。
    """
    def __init__(self, embedding_dim=1024, hidden_dim=256, dropout_rate=0.1):
        """
        初始化Router网络。
        Args:
            embedding_dim (int): 输入文本特征的维度。
                                 对于SD 2.1，这个值是1024。
                                 对于SD 1.5，这个值是768。
            hidden_dim (int): 隐藏层的维度。
            dropout_rate (float): Dropout比率，用于防止过拟合。
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 输出一个连续值（预测的复杂度分数）
        )
        print(f"Initialized PromptComplexityRouter with embedding_dim={embedding_dim}")

    def forward(self, prompt_embedding):
        """
        前向传播。
        Args:
            prompt_embedding (torch.Tensor): 形状为 (batch_size, embedding_dim) 
                                             或 (batch_size, sequence_length, embedding_dim) 的文本特征。
        
        Returns:
            torch.Tensor: 形状为 (batch_size, 1) 的预测复杂度分数。
        """
        # 兼容不同文本编码器可能输出的3维张量
        if prompt_embedding.dim() == 3:
            # 取序列维度的平均值来获得一个固定的向量表示
            prompt_embedding = prompt_embedding.mean(dim=1)
            
        return self.network(prompt_embedding)