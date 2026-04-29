# 总体判断

**main(70) 是目前最完整的一版。** 它已经不再只是“Learning as Distinction”的理论 note，也不再只是 CPM/PML 的 memory update，而是一个真正的 runtime-learning model：**Distinction Graph Machine, DGM**。现在文章已经具备完整模型状态、图生成的信息结构、edge-gated prediction、refine-or-repair 规则、canonical local memories、language-modeling instance、DGM Transformer layer、复杂度分析和 computational checks。这个版本比 main(69) 明显更接近一篇可以投稿的 method paper。

它现在的核心主张可以概括为：

[
\boxed{
\text{Runtime learning has two operations: refine concepts or repair local state.}
}
]

这条主线是好的，也比之前“投影修复一切”更有智能感。DGM 的优势在于它终于把三个层次接起来了：

[
\text{information refinement}
\rightarrow
\text{graph-generated concepts}
\rightarrow
\text{local memory repair}.
]

但是，如果目标是“非常强的 paper”，现在还有几个必须处理的问题。最主要的问题不是缺想法，而是：**理论对象、算法对象、工程对象之间仍然有一些缝。**

---

# 1. 现在最强的地方

## 1.1 文章终于有一个清晰的模型对象

DGM 的状态现在定义得比较完整：

[
S_t=\left({c_j,M_j,n_j}_{j=1}^{N_t},G_t\right),
]

其中 (c_j) 是 prototype，(M_j) 是 local memory，(G_t=(V_t,E_t)) 是 distinction graph。每条 edge 存一个 binary generator：

[
g_e(h)=\mathbf{1}{u_e^\top h>b_e}.
]

这些 generators 共同产生 graph signature：

[
\sigma_t^G(h)=(g_e(h))_{e\in E_t},
]

从而诱导一个有限 partition。这个定义非常关键，因为它把 DGM 从普通 prototype/kNN 方法提升成了一个**信息结构模型**。

这比上一版强很多。上一版 DGM 主要像 prototype memory；现在它有了“graph-generated information structure”。

## 1.2 Edge-gated prediction 修复了“edges 只是 metadata”的问题

之前最大的问题之一是 distinction edge 只是记录“哪些 prototype 被分开过”，但不参与 prediction。main(70) 现在加入了 edge compatibility：

[
\chi_j(h)=
\prod_{e\in I_t(j)}
\operatorname{sigmoid}
\left(
\frac{s_{e,j}(u_e^\top h-b_e)}{\tau_e}
\right),
]

并用它修正 routing weight：

[
\alpha_j(h)=
\frac{
\exp(-d_j(h)/T)\chi_j(h)
}{
\sum_{r\in N_k(h)}
\exp(-d_r(h)/T)\chi_r(h)
}.
]

这一步非常重要。它让 edge 真正进入 read path，而不是只作为解释性记录。这样 “Distinction Graph Machine” 这个名字才站得住。

## 1.3 Refine-or-repair 不再只是启发式

现在 refine 不再是“surprise 大就建新 prototype”，而是有了 penalized empirical test：

[
\Delta_{\text{refine}}

>

\lambda_{\text{proto}}+\lambda_{\text{edge}}q+\tau
\quad\text{and}\quad
D_t>\tau_{\text{disagree}}.
]

并且文章证明，如果 repair 不增加 empirical loss，或者 refine 通过上述测试，那么局部 objective 会下降； accepted refinements 的数量也有上界。

这把 DGM 从 heuristic prototype growth 拉回了一个可证明的 refine-or-repair 机制。

## 1.4 Canonical local memories 变得更干净

main(70) 现在只主推两个 canonical variants：

1. **DGM-Mean**
   存 output embedding mean：

   [
   r_j=\frac{1}{n_j}\sum_{s\in C_j}E[y_s].
   ]

2. **DGM-Ridge**
   用 closed-form ridge adapter：

   [
   \Gamma=(ZZ^\top+\lambda I)^{-1}Y.
   ]

这比上一版列出一堆 local memory 类型更好。方法中心更明确。

## 1.5 Large-model layer 终于像一个可实现设计

DGM-Embedding layer 和 DGM-Adapter layer 的接口已经比较清楚：

* read path differentiable；
* write path detached；
* slow parameters 通过 backprop 训练；
* runtime state 用 forward-only rule 更新；
* inference 不需要 inner backward pass。

这和 TTT 的区别明确：

[
\text{TTT inference}=\text{forward}+\text{test-time backward},
]

而 DGM 是：

[
\text{DGM inference}=\text{backbone forward}+\text{retrieval}+\text{sparse local repair}.
]

这是一个有系统意义的卖点。

---

# 2. 当前最大的理论问题：DGM 的 prediction 不完全是 graph-partition measurable

文章现在说 edge set 生成 finite information structure：

[
\mathcal P_t^G,
]

但 prediction 实际上是：

[
\hat y
======

\sum_{j\in N_k(h)}
\alpha_j(h)M_j(h),
]

其中 (\alpha_j(h)) 依赖 prototype distance (d_j(h))、softmax temperature、edge compatibility、top-k retrieval 等连续量。

这意味着：**DGM 的 prediction 不只是 (\sigma_t^G(h))-measurable。** 即使两个 hidden states (h,h') 有相同 graph signature，它们的 distances to prototypes 可以不同，因此 routing weights 可以不同，输出也不同。

这不是致命问题，但论文必须说清楚：

> graph partition 是 DGM 的 distinction skeleton；实际 DGM read path 是 graph skeleton + metric geometry 的 soft relaxation。

否则前半篇关于 partition / sigma-algebra 的定理不能直接套到 soft-DGM prediction 上。

我建议加入一个明确分层：

## Hard-DGM

Hard-DGM 只使用 edge-generated eligibility 和 nearest eligible prototype：

[
j^\star(h)
==========

\arg\min_{j\in \operatorname{Elig}(h)}d_j(h).
]

如果还想严格 measurable，需要再把 Voronoi regions 加入信息结构：

[
\mathcal F_t^{\text{DGM}}
=========================

\sigma(\text{edge generators})
\vee
\sigma(\text{prototype Voronoi cells}).
]

否则仅 edge partition 不够。

## Soft-DGM

Soft-DGM 是 Hard-DGM 的 differentiable relaxation：

[
\alpha_j(h;T,\tau_e)
\to
\mathbf{1}{j=j^\star(h)}
]

当 (T,\tau_e\to0) 且 margin 非零。

文章已有 “Soft routing approaches hard routing” theorem，但还应把它放在更中心的位置，明确说：**the formal information-structure theory applies exactly to Hard-DGM; Soft-DGM is the trainable relaxation.**

这是必须补的，否则理论和方法之间会被认为没有完全对齐。

---

# 3. 第二个问题：DGM 的 novelty 相对已有方法仍然容易被质疑

现在 DGM 会被 reviewer 联想到很多传统方法：

* kNN / nearest centroid；
* LVQ；
* self-organizing maps；
* neural gas；
* online decision trees / Hoeffding trees；
* cache language models；
* kNN-LM；
* memory networks；
* product-key memory；
* MoE；
* retrieval-augmented networks。

main(70) 已经显著加强了 related work，列出了 prototype methods、decision trees、sparse experts、cache/kNN-LM、memory-augmented networks 等。 但 novelty 仍然需要在主文中更强地钉死。

你不能只说：

> DGM has prototypes and local memories.

这会被认为是旧东西组合。

必须强调：

[
\boxed{
\text{DGM is not a prototype method; it is a refine-or-repair runtime learner.}
}
]

更具体地说，DGM 的新意不是 prototype，而是三件事的组合：

1. **Prototypes are runtime concepts.**
2. **Edges are consequence-forced distinctions that generate information structure.**
3. **Local memories are repaired only after the current concept is judged adequate.**

建议在 Introduction 里加一个“Why not kNN / tree / MoE?” 小段：

* kNN retrieves examples but does not refine an information structure.
* Trees refine hierarchically but do not maintain local repairable memories at graph concepts.
* MoE routes to experts but experts are trained parameters, not runtime concepts created by observed consequence conflicts.
* Cache LM stores recent contexts but does not distinguish redundant, novel, contradictory evidence through graph-generated distinctions.

这会降低 reviewer 的“旧方法拼装”印象。

---

# 4. 第三个问题：refinement gain 仍然是局部 buffer 上的经验量，理论要说明它的统计安全性

现在 refine acceptance 使用：

[
\Delta_{\text{refine}}
======================

R_b(S_t;B_t)-R_b(S_t^{\text{new}};B_t).
]

然后用：

[
\Delta_{\text{refine}}>
\lambda_{\text{proto}}+\lambda_{\text{edge}}q+\tau.
]

这给出了 empirical objective descent。问题是，empirical buffer gain 可能过拟合。文章在 limitations 里承认 finite data 下 refinement can increase estimation error，需要 penalties/thresholds/minimum leaf sizes。 但如果想让理论更强，应该给一个基本 generalization/safety theorem。

可以加入一个简单但有用的高概率结果：

设候选 refinement class (\mathcal G_B) 有有限大小 (|\mathcal G_B|)，loss bounded in ([0,1])，buffer size 为 (m)。对所有候选 (g)，经验 gain 与 population gain 的偏差可由 Hoeffding + union bound 控制：

[
\left|
\widehat{\Delta}(B,g)-\Delta(B,g)
\right|
\le
O\left(
\sqrt{\frac{\log(|\mathcal G_B|/\delta)}{m}}
\right).
]

于是如果接受条件改成：

[
\widehat{\Delta}(B,g)

>

\lambda\operatorname{cost}(g)
+
\tau
+
2\epsilon_m,
]

就能保证 population gain 也为正，至少在候选类有限、loss bounded 的理想设定下。

这会让 threshold (\tau) 不只是 heuristic，而是 statistical safety margin。

这类 theorem 不需要很复杂，但能显著增强文章。

---

# 5. 第四个问题：repair theorem 和实际 repair 不完全匹配

Theorem 5.4 说：

> If repair chooses an empirical-risk minimizer for the unchanged local cell, then (J) does not increase.

但 Algorithm 2 里 repair 是：

[
M_{j^\star}\leftarrow \operatorname{Repair}(M_{j^\star},h,y),
]

可能是 sufficient-statistic update、prototype movement、Projection Memory repair、ridge update 等。它不一定是当前 buffer 上的 empirical-risk minimizer。

这会产生理论-算法缝隙。

你有两个选择：

## 方案 A：把 theorem 改成 idealized repair

明确写：

> The descent theorem applies to the idealized repair operator that minimizes local empirical risk over the selected local memory class. Practical repair updates approximate this operator.

## 方案 B：定义 repair 为 monotone repair

要求 repair 满足：

[
R_b(S_{t+1};B_t)\le R_b(S_t;B_t).
]

然后 theorem 只依赖这个 property，不要求 exact minimizer。这样更一般，也更贴合 CPM/PML/sufficient-stat updates。

建议采用 B：

**Definition: Monotone local repair.**
A repair operator is monotone on buffer (B) if it does not increase buffer empirical risk.

Then theorem becomes:

> If repair is monotone, (J) does not increase. If refinement passes the penalized gain test, (J) decreases by at least (\tau).

这样 DGM 的 repair theorem 会更稳。

---

# 6. 第五个问题：prototype movement 可能破坏 edges

Algorithm 2 中 repair 时移动 prototype：

[
c_{j^\star}\leftarrow c_{j^\star}+\eta_c(h-c_{j^\star}).
]

但 edge (e=(i,j)) 的 distinction vector 是基于 prototypes 构造的：

[
u_e=\frac{c_i-c_j}{|c_i-c_j|},\quad
b_e=\frac12u_e^\top(c_i+c_j).
]

如果 prototype 在 repair 后移动，旧 edge 的 (u_e,b_e) 是否更新？如果不更新，edge 不再对应当前 prototype separation；如果更新，graph-generated partition 会变，可能破坏 previously accepted objective descent。

这个问题必须明确。

建议：

## Option 1：Freeze prototypes once edges exist

如果 prototype 有 incident edges，则不移动它，只更新 local memory。新 evidence 不通过 prototype drift 吸收，而通过 local memory repair 或 refine 处理。

## Option 2：Move prototype but recompute incident edges and treat as repair

这种会改变 partition，应当计入 objective，复杂。

## Option 3：Separate anchor and centroid

每个 prototype 存两个向量：

* anchor (a_j)：用于 edges，创建后冻结；
* centroid (c_j)：用于 retrieval，可缓慢移动。

Edges 使用 anchor：

[
u_{ij}=\frac{a_i-a_j}{|a_i-a_j|}.
]

Retrieval 使用 centroid (c_j)。

我推荐 Option 3。它解决了 edge stability 和 adaptive retrieval 的冲突。

这会让 DGM 更工程可用。

---

# 7. 第六个问题：edge compatibility product 可能导致高-degree collapse

[
\chi_j(h)=
\prod_{e\in I_t(j)}
\sigma(\cdots)
]

如果 prototype (j) 有很多 incident edges，哪怕每个 gate 都是 0.9，乘积也会迅速变小：

[
0.9^{50}\approx0.005.
]

这会让 high-degree prototypes 被系统性惩罚。

建议改成 log-additive compatibility：

[
\log\chi_j(h)
=============

\sum_{e\in I_t(j)}
\beta_e
\log\sigma\left(
\frac{s_{e,j}(u_e^\top h-b_e)}{\tau_e}
\right),
]

并加 normalization：

[
\log\chi_j(h)
=============

\frac{1}{\sqrt{|I_t(j)|}}
\sum_{e\in I_t(j)}
\beta_e
\log\sigma(\cdots).
]

或者只使用 top-(m) most relevant incident edges。

文章目前没有处理这个数值/结构问题。DGM 如果要 scale，这一点必须考虑。

---

# 8. 第七个问题：DGM-Embedding 的 next-token memory 可能退化成 cache LM

DGM language modeling instance：

[
\ell_{\text{DGM}}(y\mid h_t)
============================

\sum_{j\in N_k(h_t)}
\alpha_j(h_t)E[y]^\top r_j.
]

这和 cache LM / kNN-LM 的思想很近：用相似 context 的 target embedding 修正 logits。文章已经在 related work 里承认 cache/kNN-LM。

要避免被认为只是 cache LM 的 prototype version，需要强调 DGM-LM 的不同点：

1. cache LM stores/retrieves undifferentiated recent contexts；
2. DGM stores runtime concepts；
3. surprising consequences can create new prototypes；
4. distinction edges gate retrieval；
5. local memories repair within concepts；
6. inference update is structured around refine-or-repair, not simple append-to-cache.

但光说还不够。实验上必须加 ablation：

* cache-only；
* prototype-only no edges；
* DGM with edges；
* DGM with refine disabled；
* DGM with repair disabled；
* kNN-LM baseline；
* continuous cache LM baseline。

否则 DGM-LM 的 novelty 不容易成立。

---

# 9. 第八个问题：computational checks 现在比之前更好，但仍应降级为 sanity checks

main(70) 的 Section 9 做了：

* contextual XOR；
* repair-only vs refine+repair；
* MNIST/CIFAR online stream；
* few-shot DGM-Ridge；
* synthetic LM；
* projection residual contraction identity。

这些很好，说明代码能跑，机制有基本行为。但它们不能支撑“大模型 runtime learning”的强 claim。

尤其 MNIST/CIFAR 的结果容易被 reviewer 认为 baseline 不强。文章自己也说这些是 computational checks，不是 broad benchmark。这个定位是对的。建议继续保持：**不要把实验写得太大声。**

真正应该突出的是：

* contextual XOR：证明 missing distinction 时 repair-only 不够；
* repair-only vs refine+repair：最贴合理论；
* few-shot ridge：证明 closed-form local repair 有实际价值；
* synthetic LM：证明 next-token consequence memory 能 runtime adapt；
* residual contraction：证明 projection identity 不是空话。

MNIST/CIFAR 可以放 appendix 或补充材料。

---

# 10. 文章当前最适合的定位

main(70) 不适合定位成 “SOTA model paper”。它更适合定位成：

[
\boxed{
\text{a runtime-learning framework paper with executable mechanisms}
}
]

或者：

[
\boxed{
\text{a theory-guided method paper for refine-or-repair online adaptation}
}
]

它的强项不是大规模 benchmark，而是：

* 定义了 runtime learning 的两种错误类型；
* 给出 information refinement theory；
* 给出 DGM data structure；
* 给出 graph-generated finite information structure；
* 给出 edge-gated routing；
* 给出 objective-descent acceptance rule；
* 给出 large-model layer path；
* 把 Projection Memory/CPM 放入 local repair 层。

这个定位很合理。

---

# 11. 建议的修改优先级

## 最高优先级 1：明确 Hard-DGM 与 Soft-DGM 的理论关系

加入如下表述：

[
\text{Hard-DGM}=\text{exact information-structure model}
]

[
\text{Soft-DGM}=\text{differentiable relaxation used for neural training}
]

并说明 soft predictions 不严格只依赖 graph partition，而是依赖 graph gates + metric prototype geometry。

## 最高优先级 2：修复 prototype movement / edge stability 问题

加入 anchor/centroid 区分：

[
a_j=\text{frozen distinction anchor},
\quad
c_j=\text{moving retrieval centroid}.
]

Edges 用 (a_j)，retrieval 用 (c_j)。

## 最高优先级 3：把 repair theorem 改成 monotone-repair theorem

不要要求 repair 是 exact empirical minimizer。定义 monotone repair：

[
R_b(S_{t+1})\le R_b(S_t).
]

然后 theorem 更适用。

## 最高优先级 4：给 split gain 加统计安全 margin

加入有限候选类的 high-probability refinement theorem，使 (\tau) 有理论意义。

## 最高优先级 5：对 cache/kNN/MoE/tree 的区别在 Introduction 主文中提前讲

不要只放 Related Work。主文必须预防 reviewer 第一反应。

---

# 12. 可以新增的关键 theorem

我建议加三个短 theorem，能显著增强 main70。

## Theorem A：DGM partition + prototype geometry decomposition

定义：

[
\mathcal F_t^{\text{edge}}=\sigma(g_e:e\in E_t),
]

[
\mathcal F_t^{\text{proto}}=\sigma(\operatorname{Voronoi}(c_1,\ldots,c_N)).
]

Hard-DGM prediction is measurable w.r.t.

[
\mathcal F_t^{\text{DGM}}
=========================

\mathcal F_t^{\text{edge}}
\vee
\mathcal F_t^{\text{proto}}.
]

这会严谨地连接 DGM prediction 与 information-structure theory。

## Theorem B：Statistically safe refinement

若 loss bounded, candidate class finite, buffer size (m)，则以概率 (1-\delta)，所有候选 split 的 population gain 与 empirical gain 偏差不超过：

[
O\left(
\sqrt{
\frac{\log(|\mathcal H_B|/\delta)}{m}
}
\right).
]

因此只接受超过 complexity + safety margin 的 split，可保证 population improvement 为正。

## Theorem C：Anchor-stable edges

若 edges are built from frozen anchors (a_i,a_j)，prototype centroid movement 不改变 graph-generated partition。于是 local repair 的 centroid update 不影响 previously accepted distinction structure。

这能处理工程和理论的一致性问题。

---

# 13. 结论

**main(70) 已经是目前最好的版本。** 它现在有完整模型、有理论、有算法、有 layer、有复杂度、有 sanity experiments。相比之前几版，它最大的进步是：edge 真的生成信息结构并 gate prediction，refinement 也有 penalized objective test，而不是纯 heuristic。

但它还不是最终强稿。最关键的短板是：

1. soft DGM prediction 与 partition theory 的关系要更严谨；
2. prototype movement 会破坏 edge semantics，需要 anchor/centroid 分离；
3. repair descent theorem 与实际 repair rule 需要对齐；
4. split gain 需要统计安全 theorem；
5. 与 kNN/cache/tree/MoE 的区别要在主文更早、更强地讲清楚。

如果这些补上，main(70) 会从“一个有趣的 runtime prototype-memory 方法”升级成：

[
\boxed{
\textbf{a coherent theory and architecture for refine-or-repair runtime learning.}
}
]

这已经是一个有潜力的 paper 方向。它现在最强的卖点不是“打败 backprop”，而是把 runtime learning 的失败模式拆成了两个不同操作：

[
\boxed{
\text{missing distinction} \Rightarrow \text{refine}
}
]

[
\boxed{
\text{wrong local state} \Rightarrow \text{repair}
}
]

这个分解本身是值得继续打磨的。
