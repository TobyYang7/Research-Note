---
citeKey: "@ciniFillingG_ap_sMultivariate2022"
year: "2022"
title: "Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks"
tags:
  - paper
  - STdata
  - imputation
---

> ## TODO
>
>- [ ] code: [github](https://github.com/Graph-Machine-Learning-Group/grin)
>- [ ] pytorch-lightning
>- [ ] paper
>- [ ] discharge dataset
>- [ ] 6.13组会讲解代码框架
>
---

> ## Metadata
>
>- **Title**：Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks
>- **Author**：Andrea Cini, Ivan Marisca, Cesare Alippi
>- **Entry**：[Zotero link](zotero://select/items/@ciniFillingG_ap_sMultivariate2022) [URL link](http://arxiv.org/abs/2108.00298)
>- **Tags**:
>- **Other**：2022 - arxiv:2108.00298 -
>
---

# Note

## Grin Model

### Dataset

X: 输入的时间序列数据，形状为 (T, N, F)
M: 掩码矩阵，形状为 (T, N)
A: 邻接矩阵，形状为 (N, N)
$X_{hat}$ 是填补后的时间序列数据，形状为 (T, N, F)
图的结构是固定的

数据模块信息:
训练数据集大小: 4948
训练批次大小: 32
验证数据集大小: 564
验证批次大小: 32
测试数据集大小: 2788
测试批次大小: 32

- in-sample imputation:  训练和验证数据来自相同的时间范围
- out-of-sample imputation: 训练和验证数据来自不同的时间范围

### Motivation

- 背景：数据缺失普遍发生
- 之前的补全方法并不能很好地捕获/利用 不同sensor之间的非线性时间/空间依赖关系
- 文章中比较的几个baseline都不是ST model

### Future Work

- model假设数据是平稳的 （[[异常值检测]]？）
- 如何确保数据确实的可靠性（白+黑）
- virtual and active sensing?

![[image-20240610221501465.png]]

### Structure

- 图的结构是固定的
- bidirectional: 沿着时间轴，同时正向计算和反向计算
- spatial decoder: 第一层[[imputation]]为linear readout，第二层（MPNN）通过时空信息进行优化
- [[GRU]]基本结构：

$$
\begin{align*}
r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r) \\
u_t &= \sigma(W_u x_t + U_u h_{t-1} + b_u) \\
\tilde{h}_t &= \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h) \\
h_t &= u_t \odot h_{t-1} + (1 - u_t) \odot \tilde{h}_t
\end{align*}
$$
![[1_i-yqUwAYTo2Mz-P1Ql6MbA.webp]]

- [[recurrent]]的概念: MPGRU采用GRU的框架，通过MPNN提取空间信息，然后再通过gate的结构传递时序信息

$$
\begin{align*}
r_t^i &= \sigma(\text{MPNN}([\hat{x}_t^{(2)i} || m_t^i || h_{t-1}^i], W_r)) \\
u_t^i &= \sigma(\text{MPNN}([\hat{x}_t^{(2)i} || m_t^i || h_{t-1}^i], W_u)) \\
c_t^i &= \tanh(\text{MPNN}([\hat{x}_t^{(2)i} || m_t^i || r_t^i \odot h_{t-1}^i], W_t)) \\
h_t^i &= u_t^i \odot h_{t-1}^i + (1 - u_t^i) \odot c_t^i
\end{align*}
$$

- 在GRU的每层计算中，都会衔接GCN计算相邻节点的特征（重复提取K次）
- first stage仅仅是一个linear output，在这之后衔接MPGRU，最后再套一层spatial decoder

![alt text](<assets/Filling the G_ap_s_ Multivariate Time Series Imputation by Graph Neural Networks/image.png>)
![alt text](<assets/Filling the G_ap_s_ Multivariate Time Series Imputation by Graph Neural Networks/image-1.png>)

### 模型参数设置

```bash
GraphFiller(
  (loss_fn): MaskedMetric()
  (train_metrics): MetricCollection(
    (train_mae): MaskedMAE()
    (train_mape): MaskedMAPE()
    (train_mre): MaskedMRE()
    (train_mse): MaskedMSE()
  )
  (val_metrics): MetricCollection(
    (val_mae): MaskedMAE()
    (val_mape): MaskedMAPE()
    (val_mre): MaskedMRE()
    (val_mse): MaskedMSE()
  )
  (test_metrics): MetricCollection(
    (test_mae): MaskedMAE()
    (test_mape): MaskedMAPE()
    (test_mre): MaskedMRE()
    (test_mse): MaskedMSE()
  )
  (model): GRINet(
    (bigrill): BiGRIL(
      (fwd_rnn): GRIL(
        (cells): ModuleList(
          (0): GCGRUCell(
            (forget_gate): SpatialConvOrderK(
              (mlp): Conv2d(330, 64, kernel_size=(1, 1), stride=(1, 1))
            )
            (update_gate): SpatialConvOrderK(
              (mlp): Conv2d(330, 64, kernel_size=(1, 1), stride=(1, 1))
            )
            (c_gate): SpatialConvOrderK(
              (mlp): Conv2d(330, 64, kernel_size=(1, 1), stride=(1, 1))
            )
          )
        )
        (norms): ModuleList(
          (0): Identity()
        )
        (first_stage): Conv1d(64, 1, kernel_size=(1,), stride=(1,))
        (spatial_decoder): SpatialDecoder(
          (lin_in): Conv1d(66, 64, kernel_size=(1,), stride=(1,))
          (graph_conv): SpatialConvOrderK(
            (mlp): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
          )
          (lin_out): Conv1d(128, 64, kernel_size=(1,), stride=(1,))
        (cells): ModuleList(
          (read_out): Conv1d(128, 1, kernel_size=(1,), stride=(1,))
          (activation): PReLU(num_parameters=1)
        )
        (h0): ParameterList(  (0): Parameter containing: [torch.FloatTensor of size 64x36])
      )
      (bwd_rnn): GRIL(
          (0): GCGRUCell(
            (forget_gate): SpatialConvOrderK(
              (mlp): Conv2d(330, 64, kernel_size=(1, 1), stride=(1, 1))
            )
            (update_gate): SpatialConvOrderK(
              (mlp): Conv2d(330, 64, kernel_size=(1, 1), stride=(1, 1))
            )
            (c_gate): SpatialConvOrderK(
              (mlp): Conv2d(330, 64, kernel_size=(1, 1), stride=(1, 1))
            )
          )
        )
        (norms): ModuleList(
          (0): Identity()
        )
        (first_stage): Conv1d(64, 1, kernel_size=(1,), stride=(1,))
        (spatial_decoder): SpatialDecoder(
          (lin_in): Conv1d(66, 64, kernel_size=(1,), stride=(1,))
          (graph_conv): SpatialConvOrderK(
            (mlp): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
          )
          (lin_out): Conv1d(128, 64, kernel_size=(1,), stride=(1,))
          (read_out): Conv1d(128, 1, kernel_size=(1,), stride=(1,))
          (activation): PReLU(num_parameters=1)
        )
        (h0): ParameterList(  (0): Parameter containing: [torch.FloatTensor of size 64x36])
      )
      (out): Sequential(
        (0): Conv2d(265, 64, kernel_size=(1, 1), stride=(1, 1))
        (1): ReLU()
        (2): Dropout(p=0, inplace=False)
        (3): Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
)
```
