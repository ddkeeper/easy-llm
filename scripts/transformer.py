import torch
import torch.nn as nn
from einops import rearrange, einsum


# Problem 1: 实现 linear 模块 (1 分)
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__() # 调用父类初始化
        '''
        in_features: int  输入特征维度
        out_features: int  输出特征维度
        device: torch.device | None = None  参数存放在哪个设备上
        dtype: torch.dtype | None = None  参数的数据类型
        '''

        # torch.empty() 只分配内存，不初始化数值；用 torch.zeros, torch.ones, torch.randn 创建张量都可以，只是会做一次无意义的填充
        self.w = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype)) # nn.Linear()
        

        mean, std = 0, 2/(in_features+out_features) ** 0.5
        nn.init.trunc_normal_(self.w, mean=mean, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        对输出应用线性变换 y = xA^T
        x: [..., in_features] 任意数量的 batch 维度的输入特征
        return: [..., out_features] 线性变换后的输出
        '''
        # einsum 实现矩阵乘法，等价于 y = x @ self.w.T 或者是 y = torch.matmul(x, self.w.T)
        y = einsum(x, self.w, "... d_in, d_out d_in -> ... d_out") # 对 d_in 维度执行矩阵变换，并将该操作广播到 d_in 维度前面的所有维度上
        return y


# Problem 2: 实现 Embedding 模块 (1 分)
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None, init_std: float = 1.0):
        super().__init__()
        '''
        num_embeddings: int  词表大小，取值范围 [0, num_embeddings‑1]
        embedding_dim: int  词向量的维度，也即模型特征维度 d_model
        device: torch.device | None = None  参数存放在哪个设备上
        dtype: torch.dtype | None = None  参数的数据类型
        init_std: float = 1.0  截断正态初始化的标准差；调小（如 0.02）可以避免训练初期 loss 尖刺，属于可控的实验变量
        '''

        self.embed = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        # 截断正态初始化，标准差由 init_std 给出，截断区间 [-3*std, 3*std]
        nn.init.trunc_normal_(self.embed, mean=0, std=init_std, a=-3 * init_std, b=3 * init_std)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''
        根据 token_ids 查表取向量
        token_ids: [batch_size, seq_len]
        return: [batch_size, seq_len, embed_dim]
        '''

        # pytorch 索引操作，等价于 torch.index_select
        return self.embed[token_ids] # 用 token_ids 中的每个元素去索引 self.embed 的第 0 维，形状变化为：(batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        # id = 0 -> self.embed[0]


# Problem 3: 实现 RMSNorm (1 point)
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None, remove: bool = False):
        super().__init__()
        '''
        d_model: int  模型隐藏层的特征维度
        eps: float = 1e-5  数值稳定性极小值，防止分母为 0
        device: torch.device | None = None  参数存放设备
        dtype: torch.dtype | None = None  参数数据类型
        remove: bool = False  消融开关，置 True 时本层退化成恒等映射（保留结构与参数数量，方便对照实验）
        '''
        self.d_model = d_model
        # 可学习缩放参数 gamma，初始化为全 1
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype)) # (d_model, )
        self.eps = eps
        self.remove = remove

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        输入形状: [..., d_model] 支持任意前置维度（如 batch_size, seq_len），最后一维为特征维度
        输出形状: [..., d_model] 与输入形状完全一致
        '''
        if self.remove:   # 消融：整层不起作用，直接把输入原样传下去
            return x

        # 记录原始精度，计算时提升到 float32 保证稳定，输出时还原精度
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # 核心计算：求均方根 -> 归一化 -> 乘可学习缩放系数
        # (..., d_model) -> (..., 1)
        rms = (x.square().sum(dim=-1, keepdim=True) / self.d_model + self.eps) ** 0.5 # (..., 1)
        result = x / rms * self.g # (..., d_model) / (..., 1) * (d_model) -> (..., d_model)

        return result.to(in_dtype)


# Problem 4: 实现 RoPE (2 points)
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        '''
        theta: float  RoPE 的 Θ 值
        d_k: int  待编码的向量维度（一般指多头注意力层里 query 和 key 向量的维度）
        max_seq_len: int  输入的最大序列长度
        device: torch.device | None = None  存储 buffer 的设备
        '''
        
        assert d_k % 2 == 0, "RoPE 要求向量维度 d_k 必须是偶数"

        # 预计算频率与角度表
        # 旋转角度 angle[i][k] = i/(theta**(2k-2)/d), i 为 token 位置下标，k 为词向量内的元素两两分组的组号
        positions = torch.arange(max_seq_len, device=device) # 旋转角度的分子 i: [0, 1, 2, 3, ..., max_seq_len - 1]
        dim_indices = torch.arange(0, d_k, 2, device=device) # 2k‑2: [0, 2, 4, 6..., d_k - 2]; k: [1, 2, 3, ..., d_k//2]

        freqs = theta ** -(dim_indices / d_k) # 频率，也即旋转角度的分母 theta**(2k‑2)/d

        # 逐位置乘构造角度表（依赖广播机制）： (max_seq_len, 1) * (1, d_k//2) -> (max_seq_len, d_k//2)
        angle = positions[:, None] * freqs[None, :]

        cos_table = torch.cos(angle)
        sin_table = torch.sin(angle)

        # 把预先计算好的 cos 表和 sin 表注册为 buffer 缓存起来，方便以后重复使用
        self.register_buffer("cos_table", cos_table)
        self.register_buffer("sin_table", sin_table)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        x: (..., seq_len, d_k) 输入张量
        token_positions: (..., seq_len) 存储每个 token 的位置下标，支持位置非连续的 token 序列： (1, 3, 5, 7...)
        return: (..., seq_len, d_k) 旋转位置编码后的张量
        '''
        cos = self.cos_table[token_positions] # (..., seq_len) -> (..., seq_len, d_k//2): 使用 token 位置下标 i 对 cos 表的第 0 维取元素，取得第 i 行的 cos 序列
        sin = self.sin_table[token_positions] # (..., seq_len) -> (..., seq_len, d_k//2): 使用 token 位置下标 i 对 sin 表的第 0 维取元素，取得第 i 行的 sin 序列

        # 两两分组后执行旋转
        # 把最后一维 d_k 按「每 2 个元素为一组」切开，每组 [x_even, x_odd]，全部组的偶数元素汇总为 x1，奇数元素汇总为 x2
        x1, x2 = rearrange(x, "... (half_d xy) -> ... half_d xy", xy=2).unbind(dim=-1)  # (..., seq_len, d_k) -> (..., seq_len, d_k//2, 2) -> x1/x2: (..., seq_len, d_k//2)

        # 分别对奇数位置的元素和偶数位置的元素执行不同的旋转操作
        x1_rot = x1 * cos - x2 * sin # (..., seq_len, d_k//2)
        x2_rot = x1 * sin + x2 * cos # (..., seq_len, d_k//2)

        # 拼接回原始形状
        # x_rot = torch.concat([x1_rot, x2_rot], dim=-1) # 用 concat 导致最后一维元素的顺序会变为 (x0, x2, x1, x3)，不符合原始的 (x0, x1, x2, x3)
        x_rot = torch.stack([x1_rot, x2_rot], dim=-1) # (..., seq_len, d_k//2) -> (..., seq_len, d_k//2, 2)
        x_rot = rearrange(x_rot, "... half_d xy -> ... (half_d xy)") # (..., seq_len, d_k//2, 2) -> (..., seq_len, d_k)
        return x_rot


# Problem 5: 实现 softmax (1 point)
def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    '''
    x: 输入张量，支持任意维度
    i: int  待计算 softmax 的维度编号
    返回形状: 与输入 x 形状完全一致
    '''

    # .max() 的返回值是一个元组 (最大值张量, 最大值下标张量)，元组里的元素不可修改，列表里的元素可以修改
    x -= x.max(dim=i, keepdim=True)[0] # 假设 i = -1: x: (... dim) x.max: (... 1) -> (... dim)

    return torch.exp(x) / torch.exp(x).sum(dim=i,  keepdim=True) # 假设 i = -1: x: (... dim) x.sum: (... 1) -> (... dim)


# Problem 6: 实现缩放点积注意力 (5 points)
def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    实现缩放点积注意力
    q, k: (batch_size, ..., seq_len, d_k)
    v: (batch_size, ..., seq_len, d_v)
    mask: 可选布尔张量 (seq_len, seq_len)
          mask=True 代表该位置允许参与注意力权重计算, False 权重置零
    return: (batch_size, ..., seq_len, d_v)
    """

    # scores = q @ k.T
    scores = einsum(q, k, "... seq_len1 d, ... seq_len2 d -> ... seq_len1 seq_len2") # (..., seq_len1, d_k), (..., seq_len2, d_k) → (..., seq_len1, seq_len2)
   
    scores /= (q.shape[-1] ** 0.5)  # 使用 sqrt(d_k) 缩放，防止维度变大后内积过大导致 softmax 饱和
    
    if mask is not None:
        scores = scores.masked_fill(mask == False, -torch.inf) # 对 False 位置（对应于上三角除对角线上的元素）进行填充，填充值为负无穷，经过 softmax 后权重 ≈0
    
    scores = softmax(scores, -1) # 在最后一个维度上做 softmax

    # y = scores @ v
    y = einsum(scores, v, "... seq_len1 seq_len2, ... seq_len2 d -> ... seq_len1 d") # (..., seq_len1 seq_len2), (..., seq_len2 d_v) → (..., seq_len1 d_v)
    return y


# Problem 7: 实现因果多头自注意力 MHA (5 points)
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, rope_theta: float) -> None:
        super().__init__()
        '''
        d_model: int  Transformer 块输入维度
        num_heads: int  注意力头数
        max_seq_len: int  RoPE 最大序列长度
        rope_theta: float  RoPE 基数频率
        '''

        self.d_model = d_model
        self.d_head = d_model // num_heads   # 计算每个头的向量维度
        self.num_heads = num_heads

        self.w_qkv = Linear(d_model, 3 * d_model) # 所有头的 qkv 矩阵合并成一个大矩阵: 输出维度 out_dim = num_heads * d_head * 3

        # rope_theta>0 启用旋转位置编码；d_head为每个头维度
        self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=self.d_head, max_seq_len=max_seq_len) if rope_theta > 0 else None

        self.w_output = Linear(d_model, d_model) # 输出投影矩阵

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        '''
        :param x: 输入张量: (batch_size, seq_len, d_model)
        :param token_positions: 每个 token 的位置下标 (batch_size, seq_len)，为 None 自动生成 0~seq_len‑1
        :return: (batch_size, seq_len, d_model)
        '''

        qkv = self.w_qkv(x) # (B, S, d_model) -> (B, S, 3*d_model)
        
        q, k, v = torch.split(qkv, self.d_model, dim=-1) # (B, S, 3*d_model) -> q/k/v: (B, S, d_model)

        # 拆分多头：把 d_model 拆成 num_heads × d_head，num_heads 放到 seq 前面，匹配 rope 的代码（回顾一下 rope 对输入 x 的要求）
        # (B, S, num_heads*d_head) → (B, num_heads, S, d_head)
        q = rearrange(q, "... seq (num_heads d_head) -> ... num_heads seq d_head", num_heads=self.num_heads, d_head=self.d_head)
        k = rearrange(k, "... seq (num_heads d_head) -> ... num_heads seq d_head", num_heads=self.num_heads, d_head=self.d_head)
        v = rearrange(v, "... seq (num_heads d_head) -> ... num_heads seq d_head", num_heads=self.num_heads, d_head=self.d_head)

        seq_len = x.shape[1]

        
        if self.rope is not None: # 是否使用 rope
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device) # 未传入位置时默认连续位置 0,1,2...seq_len‑1
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 使用 tril(lower triangle) 构造下三角矩阵，下三角位置为全 1，其他位置为全 0
        mask = torch.tril(torch.ones((seq_len, seq_len), device=q.device)).bool() # 掩码矩阵 (seq_len, seq_len)：只允许看当前以及历史 token，不能看未来 token

        output_head = scaled_dot_product_attention(q, k, v, mask) # (B, n_head, S, d_head) -> (B, n_head, S, d_head)，B 和 n_head 均为批次维度，执行广播操作

        output_head = rearrange(output_head, "... num_heads seq d_head -> ... seq (num_heads d_head)") # 多头结果拼接: (B, n_heads, S, d_head) → (B, S, n_heads*d_head=d_model)
        
        output = self.w_output(output_head) # 输出层投影: (B, S, d_model) -> (B, S, d_model)
        
        return output

# 问题 8：实现逐位置前馈网络（2 分）
class FFN(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int=512, device=None, dtype=None, gated: bool = True):
        super().__init__()
        '''
        d_model: int  模型隐藏层/词向量的维度
        d_ff: int  前馈网络中间层维度，由调用方 train_llm.py 按是否门控算好后传入
        device: torch.device | None = None  参数存放设备
        dtype: torch.dtype | None = None  参数的数据类型
        gated: bool = True  消融开关，置 False 时退化成不带门控的普通 SiLU 前馈网络（即不建 w3）
        '''
        # 此处编写你的参数创建代码
        # 127 // 64 = 1,  1 * 64 = 64,
        # 这里直接采用传进来的 d_ff（调用方按 8/3 * d_model 或 4 * d_model 对齐到 64 的倍数算好）。
        # 注意别再按 8/3 * d_model 自己重算一遍：那样 --no_gated_ffn 传进来的 4 * d_model 会被丢掉，
        # 「去掉门控后参数量对齐」就失效了，消融结论会和参数量变化混在一起。
        self.d_ff = d_ff
        self.gated = gated
        self.w1, self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype), Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        if not gated:        # 消融：删掉门控分支后，参数量减少约三分之一
            self.w3 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        FFN(x) = W2(SiLU(W1*x) ⊙ W3*x)   gated=True  门控版本（SwiGLU）
        FFN(x) = W2(SiLU(W1*x))          gated=False 普通版本（plain SiLU）
        ⊙ 代表逐‑元素相乘
        x: (batch_size, seq_len, d_model)
        return: (batch_size, seq_len, d_model)
        '''
        y1 = self.w1(x)                          # (B, S, d_model) -> (B, S, d_ff)  W1 线性映射
        y13 = y1 * torch.sigmoid(y1)             # SiLU(x) = x * sigmoid(x)，用 torch.sigmoid 保证数值稳定
        if self.gated:
            # GLU 门控：SiLU 分支再与 W3 分支逐元素相乘
            y13 = y13 * self.w3(x)               # (B, S, d_ff) ⊙ (B, S, d_ff) -> (B, S, d_ff)
        return self.w2(y13)                      # (B, S, d_ff) -> (B, S, d_model)  输出投影


# 问题 9：实现 Transformer 块（3 分）
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, rope_theta: float,
                 use_rmsnorm: bool = True, pre_norm: bool = True, ffn_gated: bool = True) -> None:
        super().__init__()
        '''
        d_model: int  Transformer 块输入维度
        num_heads: int  多头自注意力头数
        d_ff: int  前馈网络中间层维度
        max_seq_len: int  RoPE 最大序列长度
        rope_theta: float  RoPE 基数频率，传 0 或负数即关闭 RoPE（NoPE 消融）
        use_rmsnorm: bool = True   消融开关，False 时把两块 RMSNorm 都换成恒等映射
        pre_norm: bool = True      消融开关，False 时改用 Post‑Norm 结构
        ffn_gated: bool = True     消融开关，False 时 FFN 去掉门控分支
        '''
        # 子层 1：因果多头自注意力 + 残差连接
        self.mha = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, rope_theta)
        self.pre_norm1 = RMSNorm(d_model, remove=not use_rmsnorm)  # 子层 1 的归一化
        # 子层 2：前馈网络 + 残差连接
        self.ffn = FFN(d_model, d_ff, gated=ffn_gated)
        self.pre_norm2 = RMSNorm(d_model, remove=not use_rmsnorm)  # 子层 2 的归一化
        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Pre‑Norm（pre_norm=True，默认）：
        y = x + MHA(RMSNorm(x))
        z = y + FFN(RMSNorm(y))
        Post‑Norm（pre_norm=False，消融）：
        y = RMSNorm(x + MHA(x))
        z = RMSNorm(y + FFN(y))
        x: (batch_size, seq_len, d_model)
        return: (batch_size, seq_len, d_model)
        '''
        if self.pre_norm:
            # part1：子层 1 = 多头自注意力 + 残差连接
            x = x + self.mha(self.pre_norm1(x))  # (B, S, d_model) -> (B, S, d_model)
            # part 2：子层 2 = 前馈网络 + 残差连接
            y = x + self.ffn(self.pre_norm2(x))  # (B, S, d_model) -> (B, S, d_model)
        else:
            # post-norm：残差相加之后再归一化
            x = self.pre_norm1(x + self.mha(x))  # (B, S, d_model) -> (B, S, d_model)
            y = self.pre_norm2(x + self.ffn(x))  # (B, S, d_model) -> (B, S, d_model)
        return y


# 问题 10：实现 Transformer 语言模型（3 分）
class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        rope_theta: float,
        use_rmsnorm: bool = True,
        pre_norm: bool = True,
        ffn_gated: bool = True,
        tie_embedding: bool = False,
        embed_init_std: float = 1.0
    ) -> None:
        super().__init__()
        '''
        d_model: int  Transformer 块输入维度
        num_heads: int  多头自注意力头数
        d_ff: int  前馈网络中间层维度
        vocab_size: int  词表大小，决定词元嵌入矩阵维度
        context_length: int  最大上下文长度，决定 RoPE 的 sin/cos 缓存长度
        num_layers: int  Transformer 块堆叠层数
        rope_theta: float  RoPE 基数频率，传 0 或负数即关闭 RoPE（NoPE 消融）
        use_rmsnorm: bool = True      消融开关，False 时移除全部 RMSNorm
        pre_norm: bool = True         消融开关，False 时改用 Post‑Norm
        ffn_gated: bool = True        消融开关，False 时 FFN 去掉门控分支
        tie_embedding: bool = False   消融/性能优化开关，True 时输出头与输入嵌入共享同一个权重矩阵
        embed_init_std: float = 1.0   嵌入层初始化标准差
        '''
        self.tie_embedding = tie_embedding
        self.embed = Embedding(vocab_size, d_model, init_std=embed_init_std)         # 词元嵌入
        self.transformerBlocks = nn.Sequential(
            *[TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta=rope_theta,
                               use_rmsnorm=use_rmsnorm, pre_norm=pre_norm, ffn_gated=ffn_gated)
              for _ in range(num_layers)] # 列表推导式
        )                                                                             # num_layers 层 Transformer 块堆叠

        '''
        # 等价写法
        self.transformerBlocks = []
        
        for i in range(num_layers):
            trans_i = TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta=rope_theta)
            self.transformerBlocks.append(trans_i)    
        
        ===forward===
        # 输入 x 按顺序通过所有块的处理
        for block in self.transformerBlocks:
            x = block(x)
            
        '''
        self.final_norm = RMSNorm(d_model, remove=not use_rmsnorm)                   # 最终层归一化
        self.llm_head = Linear(d_model, vocab_size)                                   # 输出投影/预测头
        if tie_embedding:
            # 权重共享：输出头直接复用嵌入矩阵，省下一份 vocab_size × d_model 的参数
            # 注意这里不能再对 llm_head.w 做一次独立初始化，否则会破坏"共享同一个张量"的语义
            self.llm_head.w = self.embed.embed

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        '''
        输入: (batch_size, seq_len) 整数 token IDs
        输出: (batch_size, seq_len, vocab_size) 下一个 token 的 logits
        '''
        x_embed = self.embed(tokens)                                # (B, S) -> (B, S, d_model)
        x_blocks = self.transformerBlocks(x_embed)                 # (B, S, d_model) -> (B, S, d_model)
        llm_output = self.llm_head(self.final_norm(x_blocks))      # (B, S, d_model) -> (B, S, vocab_size)
        return llm_output

