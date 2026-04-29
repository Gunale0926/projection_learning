# 总体判断

**main(69) 比 main(68) 进步很大。** 它终于从“学习是信息细化”的纯理论 note，变成了一个有模型、有状态、有 `Predict/Observe` 算法、有 Transformer layer、有复杂度、有小实验检查的系统。现在的核心对象是 **Distinction Graph Machine, DGM**：一个由 prototypes、local memories、distinction edges 组成的 runtime learner；它遵循 “refine when concepts are too coarse, repair when local state is wrong” 的原则。这个主线比之前的 partition tree 更工程化，也比单纯 CPM 更接近“可用模型”。

但现在的问题也变得更明确：

> **理论很优雅，DGM 方法也可实现，但二者之间的连接还不够严密。**

目前理论证明的是 partition / sigma-algebra refinement；DGM 实现用的是 prototype graph + soft nearest-neighbor mixture。二者不是同一个数学对象。文章需要把 DGM graph 也形式化成一个 information structure，否则 reviewer 会觉得：前半篇是信息结构理论，后半篇是 prototype memory heuristic。

---

# 1. 现在最强的地方

## 1.1 叙事终于清楚了

当前开头的叙事非常好：

[
\text{if the atom is wrong, refine; if the local state is wrong, repair}
]

这句话抓住了你一直在找的东西：不是所有错误都应该变成梯度下降或参数更新。有些错误说明当前概念太粗；有些错误只是当前概念下的局部状态错了。main(69) 通过 DGM 把这个原则变成了 prototypes + local memories + distinction edges 的状态结构。

这是现在文章最有价值的主张。它比 “Projection Learning” 更像一个智能系统的原则。

## 1.2 终于有了可执行的 runtime learner

DGM 现在有明确状态：

[
S_t=\left({c_j,M_j,n_j}_{j=1}^{N_t},G_t\right)
]

其中 (c_j) 是 prototype，(M_j) 是 local memory，(G_t) 是 distinction graph。预测时先用 encoder 得到 (h=\phi_\theta(x))，检索近邻 prototypes，再用局部 memory 混合预测。观察到后果 (y) 后，要么 repair 最近 prototype 的 local memory，要么创建新 prototype 并加 distinction edges。

这已经是一个真正能写代码的模型，不再只是理论框架。

## 1.3 DGM layer 方向是对的

第 6 节提出了 DGM Transformer layer：read path 可微，write path detached。慢参数通过 backprop 训练，runtime state 用同一个 forward-only 更新规则在训练和推理中更新。这个方向非常重要，因为它回答了“如何接入大模型”的问题。

尤其这一点很好：

[
\text{DGM inference}=\text{backbone forward}+\text{retrieval}+\text{sparse local repair}
]

它和 TTT 的差异很清楚：TTT 在 inference 中做 inner backward，DGM 在 inference 中做 retrieval + local write。

## 1.4 小实验检查让文章更像真实 paper

第 9 节加入了 computational checks，包括 contextual XOR、MNIST/CIFAR online stream、few-shot adapter、synthetic LM、projection residual contraction check。虽然这些还不是强实验，但它们让文章不再停留在“想法”。

---

# 2. 当前最大问题：理论对象和方法对象没有完全对齐

理论部分的对象是：

[
\mathcal F_{t+1}=\mathcal F_t\vee\sigma(Z_t)
]

或者 finite partition refinement：

[
\mathcal P_E={B\cap E,B\cap E^c}
]

这些是**硬信息结构**。

但 DGM 的 prediction 是：

[
\hat y=\sum_{j\in N_k(h)}\alpha_j(h)M_j(h)
]

这是一个 soft prototype mixture，不是 partition-measurable predictor。它没有严格的 atom，也没有明确的 (\mathcal F_t)。因此前面关于 measurable predictor、minimal refinement、Bayes risk monotonicity 的理论，并不能直接作用到 DGM soft graph 上。

这不是致命问题，但必须修正。你有两个选择。

## 方案 A：先定义 Hard-DGM，再把 Soft-DGM 作为 relaxation

主文应该先定义一个 hard version：

[
j^\star(h)=\arg\min_j d(h,c_j)
]

[
\hat y=M_{j^\star}(h)
]

这时 prototypes 诱导 Voronoi partition：

[
B_j={h:j=\arg\min_i d(h,c_i)}
]

于是 DGM 确实定义了一个 partition。新 prototype creation 是 partition refinement 的近似实现。

然后再说当前 soft mixture 是可微 relaxation：

[
\alpha_j(h;T)=\frac{\exp(-d_j(h)/T)}{\sum_i\exp(-d_i(h)/T)}
]

当 (T\to 0) 时，soft read 收敛到 hard nearest-prototype read。

可以加一个简单 theorem：

> If the nearest prototype is unique with distance margin (\delta>0), then the soft routing weight on the nearest prototype tends to 1 exponentially as (T\to 0).

这样理论和 DGM layer 就接上了。

## 方案 B：把 DGM graph 定义成 generator set

把每条 distinction edge (g_{ij}) 看成一个 binary generator：

[
g_{ij}(h)=\mathbf 1{u_{ij}^\top h>b_{ij}}
]

令：

[
\mathcal G_t={g_{ij}:(i,j)\in E_t}
]

每个 (h) 的 signature 是：

[
\sigma_t(h)=(g(h))_{g\in\mathcal G_t}
]

atoms 是相同 signature 的等价类。这样 DGM graph 真正生成一个 partition / sigma-algebra。然后 prototypes 和 local memories 是每个 atom 或 local neighborhood 上的统计模型。

这会让 DGM 的 “distinction edges” 真正成为理论对象，而不是只是图上的记录。

我更推荐 **方案 B + hard/soft bridge**。这样文章会更严谨。

---

# 3. 第二个问题：distinction edges 现在没有真正参与预测

现在 DGM 创建 edge：

[
u_{ij}=\frac{c_i-c_j}{|c_i-c_j|},\qquad b_{ij}=\frac12u_{ij}^\top(c_i+c_j)
]

并证明它能分开两个 prototypes。

但在 prediction algorithm 中，预测主要靠 kNN prototypes 和 soft weights。edge 只是“record which concepts had to be separated”。如果 edge 不参与 routing、gating、memory selection 或 conflict detection，它就像 metadata，不是核心机制。

这会削弱 “Distinction Graph” 这个名字。

你需要让 edge 进入算法。最简单做法：

## Edge-gated routing

对于候选 prototype (j)，如果存在 edge ((i,j))，并且 (h) 落在 (i) 一侧，则降低 (j) 的 routing score。

例如定义 compatibility：

[
\chi_j(h)=\prod_{(i,j)\in E_t}\sigma\left(s_{ij}(u_{ij}^\top h-b_{ij})/\tau_e\right)
]

其中 (s_{ij}\in{+1,-1}) 表示 (j) 应该在哪一侧。

然后 routing 改成：

[
\alpha_j(h)\propto \exp(-d_j(h)/T)\chi_j(h)
]

这样 edges 就不仅记录 distinction，而是实际控制 read path。

更简单的 hard version：

[
j\text{ is eligible iff }h\text{ lies on }j\text{'s side of all incident distinction edges.}
]

这会让 distinction edges 成为真正的信息结构。

---

# 4. 第三个问题：refine 条件太模糊

Algorithm 2 里说：

* 如果 inconsistency score (\rho=L(y,M_{j^\star}(h))) 小，则 repair；
* 如果 (\rho) 大且 neighborhood contains incompatible consequences，则 refine，创建新 prototype。

这里的 “incompatible consequences” 没有数学定义。这个地方必须 formalize，否则算法看起来像 heuristic。

建议定义三个量：

## 4.1 Local surprise

[
\rho_t=L(y_t,M_{j^\star}(h_t))
]

## 4.2 Neighborhood consequence disagreement

对近邻集合 (N_k(h_t))，定义：

分类：

[
D_t=\max_{i,j\in N_k(h_t)} \operatorname{TV}(p_i,p_j)
]

或：

[
D_t=1-p_{j^\star}(y_t)
]

回归：

[
D_t=\max_{i,j\in N_k(h_t)}|\mu_i-\mu_j|
]

语言模型：

[
D_t=-\log p_{j^\star}(x_{t+1})
]

或者 local distributions 的 KL / JS divergence。

## 4.3 Refinement gain

创建新 prototype (c_{\text{new}}=h_t) 后，比较旧局部 loss 与新局部 loss：

[
\Delta_{\text{refine}}
======================

## L_{\text{old}}(\mathcal B)

\left[
L_{\text{old}}(\mathcal B_{\text{remain}})
+
L_{\text{new}}(\mathcal B_{\text{new}})
\right]
]

接受 refine iff：

[
\Delta_{\text{refine}}>\lambda_{\text{proto}}+\lambda_{\text{edge}}q+\tau
]

这样就和前面的 empirical split / penalized objective theory 对齐。

当前 main(69) 已有 greedy split decreases penalized objective 的理论雏形，但 DGM graph refine 没有明确放进这个 objective。需要补上。

---

# 5. 第四个问题：DGM graph 和 tree 的关系说得太快

文章说 tree 是 graph 的 sparse hard-routing special case。这个说法直观上对，但数学上还没证明。

建议加一个 proposition：

## Proposition: Decision tree is a DGM with hierarchical edge constraints

给定一棵 binary distinction tree，每个 internal node 是一个 distinction (g_\nu)。构造一个 DGM，其中每个 leaf 是 prototype，edge constraints 编码 leaf path 上的 distinctions。若 routing 只允许满足 path constraints 的 prototype，并采用 hard nearest / exact leaf matching，则 DGM prediction 等价于 tree prediction。

这个 theorem 不难，但它能让 “graph generalizes tree” 这句话严谨。

---

# 6. 第五个问题：DGM local memory 类型太多，主方法不够单一

Section 5.4 说 local memory 可以是 label counts、exemplars、local mean、ridge regressor、Projection Memory matrix。

这让 DGM 显得灵活，但也会让方法中心变弱。Reviewer 会问：

> 那 DGM 到底是什么？prototype kNN？decision tree？local ridge？memory network？MoE？CPM?

你需要指定一个 canonical local memory，其他放到 appendix。

我建议主文选两个 canonical variants：

## DGM-Mean

用于分类 / LM embedding memory：

[
r_j=\frac1{n_j}\sum_{s\in C_j}E[y_s]
]

预测：

[
\ell_{\text{DGM}}(y\mid h)=\sum_{j\in N_k(h)}\alpha_j(h)E[y]^\top r_j
]

这是最简单、可扩展、可用于 LM 的版本。

## DGM-Ridge

用于 few-shot / local adaptation：

[
\Gamma=(ZZ^\top+\lambda I)^{-1}Y
]

[
\operatorname{logits}(z)=zZ^\top\Gamma
]

这个是强 adapter 版本。

Projection Memory / CPM 则作为 advanced local repair 放后面。不要把所有 local memory 都放成主选项。

---

# 7. 第六个问题：DGM Transformer layer 的 adapter 状态不够具体

第 6 节中，DGM layer 维护 prototype bank：

[
S_t={c_j,A_j,B_j,n_j}_{j=1}^{N_t}
]

read path：

[
\Delta h_t=\sum_{j\in N_k(q_t)}\alpha_j(q_t)B_jA_jh_t
]

这看起来像 sparse local LoRA / MoE adapter。然后 write path 又说可以存 embedding mean (r_j)，也可以用 Projection Memory (W_j)，或 low-rank adapter rank-one repair。

这里需要统一。

建议把 DGM layer 拆成两个明确版本：

## DGM-Embedding Layer

State:

[
{c_j,r_j,n_j}
]

Read:

[
\Delta \ell_t(y)=\sum_j\alpha_j(q_t)E[y]^\top r_j
]

Write:

[
r_j^+=\frac{n_jr_j+\alpha_jE[x_{t+1}]}{n_j+\alpha_j}
]

这非常清楚，适合 language modeling。

## DGM-Adapter Layer

State:

[
{c_j,W_j,n_j}
]

where:

[
W_j\in\mathbb R^{d\times r}
]

Read:

[
\Delta h_t=\sum_j\alpha_j(q_t)W_jK h_t
]

Write:

[
W_j^+=W_j+\frac{\eta\alpha_j}{1+\eta\alpha_j|k_t|^2}(v_t-W_jk_t)k_t^\top
]

或者 ridge version。

如果你要用 (A_j,B_j)，那应该说明：

[
W_j=B_jA_j
]

并且 write 是对 (W_j) 做 projection，而不是分别更新 (A_j,B_j)。否则闭式更新不成立。

---

# 8. 第七个问题：实验目前只能作为 sanity check，不能支撑强结论

第 9 节的 computational checks 有用，但现在的结果容易被质疑。

主要问题：

## 8.1 Online backprop baseline 太弱

“one backprop step per example” 经常不是强在线学习 baseline。需要至少比较：

* kNN / nearest centroid；
* online decision tree / Hoeffding tree；
* LVQ / prototype learning；
* RLS / online ridge；
* passive-aggressive;
* replay-buffer SGD；
* TTT-like adaptation；
* linear probe with closed-form ridge。

否则 DGM 赢 online MLP 不足以说明新机制强。

## 8.2 MNIST/CIFAR 结果不够有说服力

DGM-style memory 在 MNIST 0.851，offline MLP 0.843，online MLP 0.682。这个结果看起来不错，但 reviewer 会问：DGM 是不是只是 nonparametric exemplar / prototype method？CIFAR 结果 0.243 也不强。

更适合 DGM 的实验不是普通 image classification，而是：

* online class-incremental / context switching；
* few-shot adaptation with frozen encoder；
* long-document entity memory；
* repeated pattern / syntax regime switching；
* hidden task identity requiring distinction creation。

## 8.3 Synthetic LM 更适合，但需要扩大

Synthetic LM 结果显示 DGM-style memory 接近 offline MLP，而 online backprop 很差。这个实验更符合 DGM 的核心，因为它测试 runtime adaptation。

但需要进一步区分：

* 只 repair 不 refine；
* 只 refine 不 repair；
* refine+repair；
* DGM without edges；
* DGM with edge-gated routing；
* DGM-Ridge；
* TTT baseline；
* RLS baseline。

这样才能证明 “refine-or-repair” 不是口号。

---

# 9. 现在应该怎么改文章结构？

我建议 main(69) 改成下面结构：

## Section 1: Introduction

保留当前叙事，但更尖锐：

[
\boxed{
\text{Some errors are wrong values; some errors are missing concepts.}
}
]

## Section 2: Information refinement theory

保留 minimal refinement、contradiction、Bayes risk、log-loss/squared-loss value。

## Section 3: Distinction Graph Machines as finite information structures

这里必须 formalize DGM graph induces an information structure。

定义：

* prototypes；
* distinction edges；
* edge-generated signature；
* atoms；
* hard routing；
* soft relaxation。

## Section 4: Refine-or-repair algorithm

给出严格 algorithm：

* Predict；
* Observe；
* Repair if local loss small；
* Refine if penalized gain large；
* Create prototype；
* Create edge；
* Update local memory.

给 theorem：

[
\text{repair decreases local empirical loss}
]

[
\text{accepted refinement decreases penalized empirical objective}
]

[
\text{finite termination if }\tau>0
]

## Section 5: DGM layer for neural models

拆成：

* DGM-Embedding layer；
* DGM-Adapter layer；
* detached write；
* training/inference API；
* complexity。

## Section 6: Relation to Projection / CPM

把 CPM 作为 local repair after distinction fixed。当前 Section 7 这部分很好，可以保留。

## Section 7: Experiments

只保留强相关实验，不要把小 sanity checks 当主结果。

---

# 10. 需要新增的关键 theorem

下面这些 theorem 会让 main(69) 更闭合。

## 10.1 DGM graph generates a partition

Let (G_t) be a set of binary distinction edges (g_e). Define signature:

[
\sigma_t(h)=(g_e(h))_{e\in E_t}
]

Then:

[
h\sim_t h'
\iff \sigma_t(h)=\sigma_t(h')
]

This equivalence relation induces a partition (\mathcal P_t^G). Adding a new edge (g) updates:

[
\mathcal P_{t+1}^G=\mathcal P_t^G\vee\sigma(g)
]

This directly connects DGM to minimal refinement.

## 10.2 Edge-gated routing converges to hard partition

If soft gates use temperature (T), and a point (h) has margin (\delta) to all violated edge decisions, then the wrong-side routing mass is bounded by:

[
\le |N_k(h)|e^{-\delta/T}
]

This bridges differentiable DGM layer to hard distinction theory.

## 10.3 Refine-or-repair objective descent

Define objective:

[
J(S)=\widehat R(S)+\lambda_p N_{\text{proto}}+\lambda_e |E|
]

If repair chooses optimal local memory, (J) does not increase.

If refinement accepted only when:

[
\Delta_{\text{refine}}>\lambda_p+\lambda_e q+\tau
]

then:

[
J(S_{t+1})<J(S_t)-\tau
]

This gives DGM an actual optimization invariant.

## 10.4 Finite number of refinements

If (J\ge0) and every refinement decreases (J) by at least (\tau), then:

[
N_{\text{refine}}\le \frac{J(S_0)}{\tau}
]

This is simple but important.

## 10.5 Hard DGM is a tree generalization

Prove tree is special case of DGM with graph constrained to rooted hierarchical distinctions.

---

# 11. 当前最大 novelty 风险

DGM 现在会被拿来和以下方法比较：

* kNN / prototype classifiers；
* LVQ / neural gas / self-organizing maps；
* decision trees；
* online clustering；
* memory networks；
* mixture of experts；
* retrieval-augmented models；
* nonparametric language models；
* fast weights；
* TTT.

现在 related work 还不够。尤其 main(69) 需要加：

* prototype learning / LVQ；
* self-organizing maps / neural gas；
* online decision trees / Hoeffding trees；
* kNN-LM / cache language models；
* memory-augmented neural networks；
* product-key memory / retrieval layers；
* sparse MoE / routing;
* continual learning / dynamic expansion.

否则 DGM 会被认为“旧方法拼装”。

你要主动说：

> DGM is not merely a prototype model: prototypes define concepts, local memories repair within concepts, and distinction edges record consequence-forced separations. The novelty is the refine-or-repair decomposition and its information-theoretic semantics.

---

# 12. 最终评价

现在 main(69) 是一个**有潜力的 method paper 初稿**，比 main(68) 更接近可发表。但它还没有完全达到“工程上可实现且理论闭合”的状态。

我会给现在版本这样的评价：

* **核心思想：强。**
  Refine-or-repair 是非常好的主线。

* **理论基础：中强。**
  Minimal refinement、Bayes risk、CPM bridge 都成立，但 DGM graph 自身的理论还薄。

* **方法可实现性：中强。**
  Predict/Observe、DGM layer、language modeling instance 都有了，但关键细节仍模糊：confused neighbor、edge usage、local memory canonical form、soft/hard routing。

* **实验：弱到中。**
  现在只能算 computational checks，不足以支撑大 claim。

* **新颖性风险：中高。**
  如果不主动处理 prototype/kNN/tree/MoE/cache-LM 相关工作，会被认为是旧方法混合。

---

# 最重要的修改建议

不要再继续加很多新机制。现在需要做的是**收紧**：

[
\boxed{
\text{DGM = graph-generated information structure + local memory repair}
}
]

全文所有内容都围绕这句话。

具体三件事：

1. **让 edge 真正参与 routing。**
   否则 “Distinction Graph” 名不副实。

2. **定义 DGM graph-induced partition / sigma-algebra。**
   否则前半篇理论不能严格作用到后半篇方法。

3. **给出 refine-or-repair objective descent theorem。**
   否则 DGM algorithm 只是 heuristic，而不是理论支持的方法。

一旦这三件事补上，main(69) 会从“有想法的 prototype memory paper”变成一篇真正完整的 runtime-learning framework paper。
