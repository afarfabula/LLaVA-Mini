# 视频训练技术报告

## 0. 结论先说清楚

这份报告先纠正一个非常关键的理解偏差：

- **是的，当前实现里每个 frame 在中间的 pre-fusion 阶段，确实还会把约 `576` 个 global patch token 带进去。**
- **但这些 `576` 个 token 不是最终全部直接塞进主 LLM 做 SFT 的那一串视频 token。**
- 当前代码实际分成两层：
  - **中间 pre-fusion 工作序列**：很长，包含 `4 + 16 + 576 + 1 + 文本 token` 这一类内容
  - **最终注入主 LLM 的视频 token**：当前配置下会被切回 `compressed_image_features`，也就是 **每帧 1 个 token**，再经过 temporal pruning 默认保留 **4 帧**，所以**每个视频最终进入主 LLM 的视觉 token 大约是 `4` 个**

所以要分清：

1. **中间视觉-文本融合阶段的序列很长**
2. **最终送进主 LLM 的视觉 token 数并不大**

当前显存大，不是因为“最终主 LLM 直接吃了 8 帧 x 598 token”，而主要是因为：

- 每帧都要先构造很长的 pre-fusion 序列并跑一遍融合层
- 这个过程要重复 8 次
- 主 LLM 仍然是完整 8B 级别模型，且训练图要保留到可训练多模态模块
- plain DDP 每张卡都有完整模型副本

当前真实运行观测（worker `3759941`）：

- GPU 0: `68836 MiB / 97871 MiB`, `100%`
- GPU 1: `68864 MiB / 97871 MiB`, `99%`

## 1. 当前视频训练的正向 / 反向流程

### 1.1 数据加载与视频抽帧

代码入口在：

- `llavamini/train/train.py`

视频样本在 `__getitem__()` 里走 `load_video_org(..., num_frm=8)`：

- 使用 `decord.VideoReader`
- 按整段视频均匀抽帧
- 当前训练路径固定使用 **8 帧**
- 每帧 resize 到 `336 x 336`
- 若 `resolution_ratio=N`，则一帧会切成 `N x N` 个图块再送入视觉塔

当前真实数据训练配置是：

- `resolution_ratio = 1`
- `compressor_size = 1`
- `prefusion_layer_num = 1`

因此当前配置下：

- 每个视频 = **8 帧**
- 每帧 = **1 张图** 进 CLIP
- 没有做更激进的高分辨 patch 切块

### 1.2 每帧在模型里的处理流程

核心函数在：

- `llavamini/model/llavamini_arch.py` 里的 `encode_images_mini()`

每帧处理大致如下：

1. CLIP vision tower 编码图像
2. 得到 patch-level visual features
3. 经过 `mm_projector`
4. 通过 `compressor` 得到压缩特征
5. 通过 `_build_pyramid_outputs()` 得到：
   - `4` 个 anchor token
   - `16` 个 buffer token
6. 把这些东西和文本 embedding 拼在一起，过 `prefusion_layers`
7. 最后**不是把整串大序列返回**，而是只切回 `compressed_image_features`

### 1.3 当前到底是不是理想的 16 token / 4 token 金字塔压缩

答案是：

- **结构上有** `4 + 16`
- **但当前实现不是“只保留 16+4 token 的纯金字塔输入”**

原因是，在 `encode_images_mini()` 里，中间 pre-fusion 序列是这样拼的：

```python
x = torch.cat([
    anchor_image_features,
    buffer_image_features,
    global_image_features,
    compressed_image_features,
    text_embedding,
], dim=1)
```

也就是说，当前中间序列里包含：

- `4` 个 anchor token
- `16` 个 buffer token
- `global_image_features`
- `compressed_image_features`
- `text_embedding`

### 1.4 “576 个 token”到底在哪一层

这个问题必须说得非常精确。

在当前配置下：

- 输入分辨率 `336 x 336`
- CLIP patch size 是 `14`
- 所以空间 patch 数量约为：
  - `24 x 24 = 576`

所以：

- **每帧在中间 pre-fusion 阶段，确实会带 `576` 个 global patch token**

如果按当前配置估算，每帧中间 pre-fusion 的视觉部分大致是：

- `4` anchor token
- `16` buffer token
- `576` global patch token
- `1` compressed token（因为 `compressor_size = 1`）

也就是视觉 token 大致：

- **`4 + 16 + 576 + 1 = 597` 个视觉 token / frame**

再加上 `text_embedding`，中间 `x` 的总长度还会再加上当前文本 token 长度。

所以更准确的说法是：

- **每帧中间 pre-fusion 序列长度 = `597 + 当前文本长度`**

### 1.5 那 8 帧最后到底有多大的输出

这里是最容易误解的点。

`encode_images_mini()` 末尾并没有把整个 `x` 返回给主 LLM，而是切回：

- `compressed_image_features`
- `fusion_text_features`

并且视频路径里：

- 8 帧会分别调用 `encode_images_mini()`
- 每帧拿到的是 `frame_image_features`
- 在当前配置下，`compressor_size = 1`，所以 **每帧只返回 1 个压缩视觉 token**

之后代码会把 8 帧堆起来：

```python
stacked_frame_features = torch.stack(image_features_list, dim=1)
pruned_image_features, _ = self._apply_temporal_pruning(stacked_frame_features)
```

`_apply_temporal_pruning()` 默认：

- `keep_frames = 4`

所以当前路径实际是：

- 原始帧数：`8`
- 每帧返回给主路径的压缩 token 数：`1`
- temporal pruning 后保留帧数：`4`

因此，**最终注入主 LLM 的视频视觉 token 数大约是：**

- **`8 帧 -> 每帧 1 token -> pruning 后保留 4 帧 -> 最终约 4 个视觉 token`**

所以现在真正的情况是：

- **中间 pre-fusion 很重**
- **最终主 LLM 输入里的视频 token 反而很少**

这也是当前实现的一个关键特征：

- 显存和耗时，更多花在“每帧中间融合计算”上
- 不完全是花在“最终给 LLM 的长视频 token 序列”上

### 1.6 是否存在召回 / 路由

**有。** 当前实现里有两类机制。

#### A. buffer retrieval

代码中有：

- `buffer_query`
- `buffer_retriever`
- `buffer_retriever_norm`

这条路径会用 16 个 learnable query 去从视觉特征中做 attention 检索，得到 16 个 buffer token。

所以：

- **有召回/检索式设计**

#### B. temporal router

代码中有：

- `temporal_router`
- `_apply_temporal_pruning()`

它会对 8 帧做打分，然后默认保留 top-k 帧：

- 默认 `keep_frames = 4`

所以：

- **有时间维度的路由 / 裁剪**

但要注意一个现实问题：

- temporal pruning 是在 **每帧都已经过了一遍 `encode_images_mini()` 之后** 才做
- 所以它减少的是“送给主 LLM 的帧数”
- 不是“减少前面 CLIP + pre-fusion 的帧级计算成本”

### 1.7 反向传播怎么走

最终模型 forward 仍然走：

- `LlavaMiniLlamaForCausalLM`
- 内部委托给 Hugging Face causal LM 路径

反向传播链路是：

- loss
- 经过主 LLM 计算图
- 回传到多模态模块
- 当前真正 trainable 的模块包括：
  - `mm_projector`
  - `compressor`
  - `prefusion_layers`
  - `temporal_router`
  - `buffer_query`
  - `buffer_retriever`
  - `buffer_retriever_norm`

当前训练策略下：

- LLM backbone：**Frozen**
- vision tower：**Frozen**
- 多模态压缩/融合模块：**Trainable**

## 2. 当前 iter 速度与稳定性的主要卡点

### 2.1 之前 GPU 利用率低的主要原因

此前低利用率的主要原因有三类。

#### A. CPU 侧视频解码和预处理

每条视频样本都要做：

- `decord.VideoReader`
- 选 8 帧
- 解码帧
- resize
- image processor preprocess

如果 `dataloader_num_workers = 0`，这些都串行发生在主训练进程里，GPU 会等数据。

#### B. 每帧单独走一遍 vision tower

当前视频路径不是把 8 帧 batch 化后一次进 CLIP，而是：

- for frame in 8 frames:
- 每帧单独跑一次 `encode_images_mini()`

这会造成：

- Python 循环开销
- 更多同步点
- 更碎的 kernel launch

#### C. gloo DDP

当前为了稳定性，DDP backend 用的是：

- `gloo`

它能跑通，但不是最适合 GPU 训练的高效 backend。

### 2.2 当前这条真实数据 run 的最新事实

这里要更新一下结论。

在当前真实数据 epoch 训练上，我实时查到：

- GPU 0：`68836 MiB / 97871 MiB`, `100%`
- GPU 1：`68864 MiB / 97871 MiB`, `99%`

所以：

- **当前这条真实数据训练并不是“GPU 利用率低于 30%”的状态**
- 至少在这条 run 上，GPU 利用率已经很高了

这说明：

- 之前低占用更像是 smoke / 启动阶段 / 数据没喂满时的问题
- 现在真实数据持续训练时，GPU 已经被喂起来了

### 2.3 现在真正的速度瓶颈是什么

虽然利用率高了，但 step 时间还是偏慢，主要原因是：

- 8 帧逐帧解码
- 8 帧逐帧跑 `encode_images_mini()`
- 每帧中间 pre-fusion 序列仍然很长（`597 + 文本长度`）
- `gloo` 通信效率不如 `nccl`

所以当前更准确的结论是：

- **低 GPU 利用率不是这条真实数据 run 的主矛盾了**
- **中间 pre-fusion 太重 + 每帧串行处理 + gloo 同步** 才是当前 step 慢的主矛盾

## 3. 当前 loss 怎么来，SFT 到底怎么监督，grad_norm 怎么看

### 3.1 loss 是怎么来的

当前 loss 是标准的：

- **causal language modeling cross entropy**

训练数据是 `conversations` 格式：

- human: prompt（包含 `<video>`）
- gpt: answer

代码会把它们 tokenize 成：

- `input_ids`
- `labels`

其中：

- human prompt 位置的 label 会被 mask 成 `IGNORE_INDEX = -100`
- 所以这些位置**不计入 loss**
- 只有 assistant answer 的 token 会算 loss

### 3.2 SFT 具体在监督什么

SFT 监督的是：

- 输入：`视频 token + prompt token`
- 目标：assistant 的答案 token 序列

本质上它学的是：

- `p(answer_t | video, prompt, previous_tokens)`

所以不是“做一个视频分类 loss”，而是：

- **条件生成式监督**
- 用 teacher forcing 的方式逐 token 学 assistant 输出

### 3.3 目前 loss 怎么看

当前真实数据训练日志里，loss 已经从更高值逐步进入 `2 ~ 4` 的区间，最近一段像这样：

- `2.9454`
- `2.9837`
- `2.0894`
- `3.4917`
- `2.5793`
- `1.6465`
- `3.9363`

这个趋势从 SFT 角度看是正常的。

### 3.4 grad_norm 什么范围算正常

没有一个绝对固定的“正常范围”，因为它取决于：

- batch size
- token 长度
- learning rate
- precision
- 可训练参数量
- 是否做梯度裁剪

当前训练配置里：

- `max_grad_norm = 0.5`

这意味着：

- 即使 raw `grad_norm` 比较大，最终更新也会被裁剪

从工程经验看：

- 大多数 step 在 `10 ~ 50` 左右：通常正常
- 偶发上到 `100+`：不一定异常，要结合 loss 是否稳定看
- 如果长期几百上千并伴随 loss 爆炸：才明显危险

当前真实 run 最近很多 step 都在：

- `10 ~ 40`

这比早期 startup 阶段健康很多。

## 4. 当前 DDP 在做什么并行，能不能迅速上 8 卡

### 4.1 当前 DDP 是什么并行

当前是标准：

- **数据并行（DDP）**

也就是：

- 每张卡一份完整模型
- 每张卡吃不同 batch shard
- 每步做梯度同步

不是：

- tensor parallel
- pipeline parallel
- FSDP / ZeRO sharding

### 4.2 当前为什么用 gloo

因为 earlier worker 上：

- `nccl` 路径在 DDP 初始化时不稳定

所以后来做了显式 backend 控制，目前稳定可跑的是：

- `gloo`

### 4.3 8 卡能不能迅速适配

代码层面：

- **可以很快适配**
- 现在就是标准 `torchrun`
- 从 `2 GPU` 切到 `8 GPU`，启动方式上是直接可扩的

但工程层面：

- 如果继续用 `gloo`
- 8 卡同步成本会更重
- 数据管线也更容易变成新瓶颈

所以正确说法是：

- **8 卡在 launcher 层可以很快适配**
- **但如果 backend 仍是 gloo，不是一个理想的长期高吞吐方案**

## 5. 为什么每张卡现在会占到 68864 MiB 显存

这是当前最关键的问题之一。

### 5.1 当前观测值

当前实时观测：

- GPU 0：`68836 MiB`
- GPU 1：`68864 MiB`

### 5.2 当前参数规模

我直接根据当前 run 里的 `model_parameters_info.txt` 做了统计：

- Trainable params: `965,888,513`
- Frozen params: `8,333,768,704`
- Total params: `9,299,657,217`

按 FP16 等价粗略看：

- trainable params：约 `1.799 GiB`
- frozen params：约 `15.523 GiB`
- total params：约 `17.322 GiB`

这说明第一件事：

- **68.8 GiB 绝对不只是模型权重本身**

### 5.3 68.8 GiB 具体由什么构成

当前单卡显存大致由下面几部分叠加而成。

#### A. 每张卡都有完整模型副本

因为当前是 plain DDP：

- 每张卡都放一整份模型

包含：

- Frozen LLaMA backbone
- Frozen CLIP vision tower
- Trainable multimodal 模块

这一项本身就是一个很高的常驻底座。

#### B. 训练态 optimizer / grad / state

虽然 trainable 只有 `~0.966B` 参数，但训练态显存不只参数本身，还包括：

- 参数
- 梯度
- optimizer first moment
- optimizer second moment
- 某些 precision 路径下还可能有额外副本

粗略量级上，trainable 部分自己就很容易到：

- `~10 GiB` 甚至更多

#### C. 冻结 8B LLM 的激活内存

这是最容易被低估的部分。

虽然 LLM backbone 是 frozen，但因为 loss 要回传到 trainable multimodal 模块，所以：

- 主 LLM 的前向计算图并不是“完全不用留”
- 仍然会为 backward 保留大量 activation

这部分通常会远大于你想象中的“只有 trainable params 才占显存”。

#### D. 每帧中间 pre-fusion 序列很长

这里是核心。

虽然最终主 LLM 注入的视频 token 很少（当前约 4 个），但是：

- **每帧在 pre-fusion 阶段都会先构造一个很长的中间序列**

当前配置下，每帧中间视觉 token 约为：

- `4` anchor
- `16` buffer
- `576` global patch
- `1` compressed

合计：

- **约 `597` 个视觉 token / frame**

再加上文本 token 后：

- 每帧 pre-fusion 实际序列长度 = `597 + 文本长度`

而且这个过程要：

- 对 8 帧重复 8 次

所以显存不是花在“最后给主 LLM 的 4 个视觉 token”上，而是很大一部分花在：

- **前面 8 次 frame-level pre-fusion 的中间激活与计算图上**

#### E. DDP bucket 和 CUDA allocator cache

另外还会有一些显存来自：

- DDP gradient buckets
- 通信 buffer
- CUDA allocator cache / fragmentation

这不是最大头，但会进一步往上抬显存。

### 5.4 为什么会出现“最终只注入 4 token，但显存还是 68.8GB”

这就是现在实现最容易误解的地方。

答案是：

- **最终注入主 LLM 的视频 token 少，不代表中间没有做大计算**
- 当前设计属于：
  - 先做重的 frame-level pre-fusion
  - 再把结果压回少量 token

所以显存高的主要原因不是：

- “主 LLM 最后吃了太长的视频 token 序列”

而是：

- “每帧前面已经做了很重的融合计算，并且要保留训练图”

### 5.5 对 68864 MiB 的最终判断

当前每卡 `~68.8 GiB` 的主要来源是：

1. 每卡完整 9.3B 模型副本
2. `~0.966B` trainable multimodal 参数的 optimizer / grad state
3. 冻结 8B LLM 的 activation memory
4. 每帧中间 pre-fusion 序列很长（`597 + 文本长度`），且要重复 8 次

所以更准确的总结是：

- **当前显存大头，不是最终注入主 LLM 的视频 token 数，而是前面每帧中间融合链路太重。**

## 6. 一页纸总结

### Q1. 现在每个 frame 还把 576 个 token 都加入到 SFT 输入了吗？

- **加入到了中间 pre-fusion 序列里**
- **不是全部原样加入到最终主 LLM 的视频输入里**

### Q2. 8 个 frame 最终到底是多大的输出？

当前配置：

- `compressor_size = 1` => 每帧最终压成 `1` 个视觉 token
- `temporal_pruning_keep_frames = 4` => 8 帧里保留 4 帧

所以最终主 LLM 接到的视频 token 大约是：

- **`4` 个视觉 token / 视频**

### Q3. 当前是不是理想的 16 + 4 金字塔压缩？

- **结构上有 16+4**
- **但不是 strict 16+4-only 输入**
- 因为中间仍保留了 full global patch tokens

### Q4. 当前有没有召回路由？

- **有**
- `buffer_retriever` + `temporal_router`

### Q5. 当前每个视频多少帧？

- **8 帧**

### Q6. 当前 DDP 是什么并行？

- **标准数据并行 DDP**

### Q7. 8 卡能否快速适配？

- **代码层面可以**
- **工程上如果继续用 gloo，不是理想高吞吐方案**

## 7. 如果你要自己改，具体改哪里

下面这一节专门回答“如果当前行为和你的预想不一致，代码应该改哪里”。

### 7.1 你如果想改“每个视频抽多少帧”

改这里：

- 文件：`llavamini/train/train.py`
- 函数：`load_video_org()`
- 关键调用点：
  - `video_frames = self.load_video_org(..., num_frm=8)`

当前行为：

- 写死按 `num_frm=8` 抽帧
- 抽帧逻辑是均匀采样

你应该改的点：

1. 如果只是想改帧数：
   - 把调用处的 `num_frm=8` 改成你想要的值
2. 如果想改采样策略：
   - 改 `load_video_org()` 内部的 `get_seq_frames()` 逻辑
   - 例如改成：头中尾采样、局部密集采样、关键帧采样等

对应位置：

- `llavamini/train/train.py`
- 关键函数：
  - `load_video_org()`
  - `get_seq_frames()`

### 7.2 你如果想改“压缩后每帧保留几个 token”

改这里：

- 文件：`llavamini/model/llavamini_arch.py`
- 位置：构建 `compressor` 的地方

当前行为：

- `compressor=Resampler(grid_size=self.compressor_size, ...)`
- `Resampler` 的输出 token 数 = `grid_size^2`

所以：

- 当前 `compressor_size = 1`
- 每帧最后 `compressed_image_features` 只有 `1` 个 token

你应该改的点：

1. 如果你想每帧保留 `4` 个压缩 token：
   - 把 `compressor_size` 设成 `2`
2. 如果你想每帧保留 `16` 个压缩 token：
   - 把 `compressor_size` 设成 `4`

当前参数入口：

- 文件：`llavamini/train/train.py`
- 配置注入：
  - `model.config.compressor_size = model_args.compressor_size`

所以你可以两种方式改：

- 改启动参数里的 `--compressor_size`
- 或者改 `ModelArguments` 默认值

### 7.3 你如果想改“4 anchor + 16 buffer”这个结构

改这里：

- 文件：`llavamini/model/llavamini_arch.py`
- 函数：`_build_pyramid_outputs()`
- 参数定义位置：
  - `self.buffer_query = nn.Parameter(torch.randn(16, ...))`

当前行为：

- anchor token 固定取：`global_image_features[:, :4, :]`
- buffer token 固定是 `16` 个 query 检索出来的结果

你应该改的点：

1. 如果你想改 anchor token 数量：
   - 改 `_build_pyramid_outputs()` 里 `:4`
2. 如果你想改 buffer token 数量：
   - 改 `buffer_query` 的长度 `16`
   - 同时注意相关 shape 会跟着变

也就是说：

- `4` 是写死在 `_build_pyramid_outputs()` 里的
- `16` 是写死在 `buffer_query` 初始化里的

### 7.4 你如果想改“为什么中间还保留 576 个 global patch token”

这部分最关键，改这里：

- 文件：`llavamini/model/llavamini_arch.py`
- 函数：`encode_images_mini()`
- 关键代码：

```python
x = torch.cat([
    anchor_image_features,
    buffer_image_features,
    global_image_features,
    compressed_image_features,
    text_embedding,
], dim=1)
```

当前行为：

- 这里显式把 `global_image_features` 拼进去了
- 对当前配置来说，`global_image_features` 约等于 `576` 个 patch token / frame
- 所以虽然最终返回给主 LLM 的是压缩 token，但中间 pre-fusion 还是会先处理这 `576` 个 token

如果你想让行为更接近你预期的“真压缩”，你可以改：

1. **最直接的改法**：
   - 把 `global_image_features` 从这个 `torch.cat()` 里拿掉
2. **保守一点的改法**：
   - 不保留全部 `global_image_features`
   - 只保留它的一个子集，比如 top-k / pooled tokens
3. **如果你要 strict 16+4-only**：
   - 让这里的 `x` 只保留：
     - `anchor_image_features`
     - `buffer_image_features`
     - `compressed_image_features`（看你要不要）
     - 少量必要文本 token

这一段就是“当前行为和你预期不一致”的最核心代码位置。

### 7.5 你如果想改“最终主 LLM 到底吃几个视觉 token”

改这里：

- 文件：`llavamini/model/llavamini_arch.py`
- 函数：`encode_images_mini()`
- 函数：`prepare_inputs_labels_for_multimodal()`

当前行为分两步：

1. `encode_images_mini()` 最后返回的是：
   - `compressed_image_features`
   - `fusion_text_features`
2. `prepare_inputs_labels_for_multimodal()` 对视频：
   - 先收集 8 帧的 `frame_image_features`
   - 再做 `_apply_temporal_pruning()`
   - 再把 pruning 后的结果插入到主 LLM 的输入里

你如果想让最终主 LLM 吃更多视觉 token，有两条路：

1. 增大每帧压缩 token 数：
   - 改 `compressor_size`
2. 增大保留帧数：
   - 改 `temporal_pruning_keep_frames`

### 7.6 你如果想改“temporal pruning 现在保留 4 帧”

改这里：

- 文件：`llavamini/model/llavamini_arch.py`
- 函数：`_apply_temporal_pruning()`
- 当前默认：

```python
keep_frames = getattr(self.config, 'temporal_pruning_keep_frames', 4)
```

当前行为：

- 没显式配置时，默认保留 `4` 帧

你应该改的点：

1. 如果只是想改默认值：
   - 直接改这里的 `4`
2. 如果想把它做成训练参数：
   - 在 `ModelArguments` 或 `TrainingArguments` 增加字段
   - 在 `train.py` 里写入 `model.config.temporal_pruning_keep_frames`

### 7.7 你如果想改“最终视频 token 是怎么插进 LLM 输入的”

改这里：

- 文件：`llavamini/model/llavamini_arch.py`
- 函数：`prepare_inputs_labels_for_multimodal()`

当前行为：

- 它会找到文本里的 `IMAGE_TOKEN_INDEX`
- 然后把视频对应的 `image_features` 插入到这个位置
- 对视频来说，这里的 `image_features` 实际上就是：
  - `8` 帧逐帧压缩后
  - 再 temporal pruning 后保留下来的结果

所以如果你要改“视频 token 到底以什么形式进主 LLM”，最终一定要看这个函数。

### 7.8 你如果想改“现在为什么每帧都要走一遍重的 pre-fusion”

改这里：

- 文件：`llavamini/model/llavamini_arch.py`
- 函数：`prepare_inputs_labels_for_multimodal()`

当前视频路径是：

```python
for frame_idx in range(temporal_len):
    frame_image_features, frame_text_features = self.encode_images_mini(...)
```

当前行为：

- 每帧单独跑一次 `encode_images_mini()`
- 这意味着：
  - 每帧都单独过 CLIP
  - 每帧都单独构造长的 pre-fusion 序列
  - 每帧都单独过 `prefusion_layers`

如果你想优化成“8 帧一起算”，这一段就是你要重构的地方。

### 7.9 你如果想改“哪些模块在训练，哪些模块是冻结的”

改这里：

- 文件：`llavamini/train/train.py`
- 关键位置：
  - `if model_args.tune_mm_mlp_adapter:` 下面那一大段

当前行为：

- 先 `model.requires_grad_(False)`
- 再手动把这些模块打开：
  - `mm_projector`
  - `prefusion_layers`
  - `compressor`
  - `temporal_router`
  - `buffer_query`
  - `buffer_retriever`
  - `buffer_retriever_norm`

如果你要调整“到底训哪些”，这里就是总开关。

### 7.10 推荐你优先看的代码顺序

如果你准备自己改，我建议按这个顺序看：

1. `llavamini/train/train.py`
   - `load_video_org()`
   - `__getitem__()` 里 `num_frm=8`
2. `llavamini/model/llavamini_arch.py`
   - `encode_images_mini()`
   - `_build_pyramid_outputs()`
   - `_apply_temporal_pruning()`
   - `prepare_inputs_labels_for_multimodal()`
3. `llavamini/train/train.py`
   - trainable / frozen 模块设置

### 7.11 如果你的目标是“让当前行为更接近你的预想”，最可能要改哪三处

如果你的预想是：

- 不要中间保留那么多 patch token
- 真正做更激进的压缩
- 最终 token 路径和中间计算路径一致

那最关键的 3 个点就是：

1. `encode_images_mini()` 里这段 `torch.cat()`
   - 决定中间到底保留哪些 token
2. `compressor_size`
   - 决定每帧最终压缩 token 数
3. `_apply_temporal_pruning()`
   - 决定 8 帧最后留下几帧

这三个点一起决定了：

- 中间算多重
- 最后喂进 LLM 多大
- 显存为什么这么高
