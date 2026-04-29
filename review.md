我对 **main(75)** 的判断是：这版已经明显从“自创概念”推进到“可以和领域内已有路线对话”的阶段。它不再只是提出 DGM，而是在引言中主动把 DGM 和 ART、prototype methods、online trees、kNN/cache、fast weights、TTT/TTA、MoE 等方向作了对照；同时主方法也已经有 anchors、movable centroids、local memories、distinction edges、edge-gated routing、sample-split refinement、DGM-Embedding、optional DGM-Adapter、CPM bridge 等完整结构。

但现在的问题也更清楚：**这篇已经不缺“想法”，而是需要把主贡献压得更锋利，把和已有工作的边界讲得更可信，并修补若干算法—理论之间的接口。**

---

## 1. 这版最大的进步

这版最重要的变化是：你终于把 DGM 的核心定位讲清楚了。

以前的版本容易被看成：

[
\text{prototype memory} + \text{tree split} + \text{cache} + \text{adapter}
]

现在主线已经变成：

[
\boxed{
\text{runtime learning}=\text{refine-or-repair}
}
]

也就是：

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

这个分解是整篇最有价值的思想。它不是单纯的 kNN、cache、tree、MoE 或 TTT，而是在说：**runtime learning 的错误类型不同，因此学习操作也应该不同。**

你现在的 abstract 和 introduction 已经基本抓住了这个点：minimal refinement 是使 evidence measurable 的最小信息结构更新；contradiction inside an atom 不能由当前 measurable predictor 解决；refinement 的价值在 log loss 下是 conditional mutual information，在 squared loss 下是 conditional variance reduction。

这部分理论叙事是强的。

---

## 2. Related work 的方向对了，但还不够“压迫性”

你现在已经把主要邻近方法放进引言表格里，包括 ART、prototype methods、online trees、kNN/cache、fast weights、TTT/TTA、MoE。这个很重要，因为 DGM 确实会被这些方向包围。

但目前表格仍然偏“声明式”，例如：

| Method family | What it lacks relative to DGM                |
| ------------- | -------------------------------------------- |
| kNN/cache     | no refine-or-repair rule                     |
| online trees  | no graph concepts/local repair memory        |
| ART           | no graph-generated hidden-state memory layer |

这些说法方向是对的，但还不够让 reviewer 信服。需要更具体地说明：

* **ART** 和 DGM 都处理 stability-plasticity、category creation、mismatch 后新增类别。DGM 的区别不只是“有 graph memory”，而是：DGM 的 edge 是 consequence-forced binary generator，进入 routing，并诱导 information structure；ART 的 vigilance/resonance 是 category match 机制，不是 graph-generated measurable structure。
* **Online decision trees** 和 DGM 都做 refinement。DGM 的区别不是“树是 hierarchy、DGM 是 graph”这么简单，而是：DGM 的节点有 local repair memory，edge 是 pairwise consequence distinction，refinement 可以局部增图而不重写 root-to-leaf path。
* **kNN/cache LM** 和 DGM-Embedding 极接近。DGM 的差异必须落在：cache 追加上下文，DGM 维护 runtime concepts；cache 只 retrieve，DGM 在 consequence conflict 下 create concept and edges。
* **Fast weights / linear attention** 和 CPM/PML 接近。DGM 的差异应该是：fast weights 解决 local repair；DGM 先判断是否 missing distinction，只有 representation adequate 时才 repair。
* **TTT/TTA** 是 runtime adaptation 主线。DGM 的差异是：TTT 问 test-time 优化什么 surrogate loss；DGM 问当前错误是 missing distinction 还是 stale local memory。

建议把 related work 从单纯表格扩展成一小节“Closest ancestors and non-novel components”。主动承认：DGM 的部件都不是凭空出现，prototype、cache、tree、fast weights、online update 都有前史；DGM 的贡献是把它们放进 **refine-or-repair** 的统一 runtime learning decomposition。

---

## 3. 现在 DGM 的形式已经比较完整

当前 DGM 状态是：

[
S_t=\left({a_j,c_j,M_j,n_j}_{j=1}^{N_t},G_t\right)
]

其中：

* (a_j)：frozen distinction anchor；
* (c_j)：movable retrieval centroid；
* (M_j)：local memory；
* (G_t=(V_t,E_t))：distinction graph。

这个 anchor / centroid 分离非常关键。它解决了前面版本的一个大问题：如果 prototype 既定义 edge 又参与 repair movement，那么 repair 会改变 information structure。现在 anchor 固定，centroid 只作为 retrieval statistic，理论上更干净。

edge-gated routing 现在也有实际作用：

[
\alpha_j(h)=
\frac{
\exp(-d_j(h)/T)\chi_j(h)
}{
\sum_{r\in N_k(h)}
\exp(-d_r(h)/T)\chi_r(h)
}
]

其中 (\chi_j(h)) 来自 incident distinction edges。这个设计比“graph 只是 metadata”强很多，因为 edge 真的进入 prediction path。

这是当前稿件最重要的工程化进步。

---

## 4. 仍然有一个理论—算法缝隙：soft retrieval 的候选召回问题

理论上，Hard-DGM 的 routing 是由 edge + anchor-order 形成的 information structure。工程上，Algorithm 1 使用 centroids 做 top-(k) retrieval，然后再 edge-gate。

这会产生一个重要问题：

> 如果真正 edge-compatible 的 prototype 没有进入 (N_k(h))，edge gate 再好也没有机会选择它。

也就是说，soft-DGM 的 correctness 不仅依赖 edge-gating，还依赖 retrieval candidate recall。

你需要明确写出一个 assumption 或 theorem：

**Candidate recall condition.**
如果 Hard-DGM 的目标 prototype (j^\star) 满足：

[
j^\star \in N_k(h),
]

那么 soft routing 在 (T,\tau_e\to 0) 时收敛到 hard routing。否则没有保证。

工程上可以用一个更稳的 candidate set：

[
N(h)
====

N_k^{\text{centroid}}(h)
\cup
N_k^{\text{anchor}}(h)
\cup
\operatorname{GraphNeighbors}(N_k^{\text{centroid}}(h)).
]

这可以显著减少“正确 prototype 没被检索出来”的问题。否则 reviewer 会指出：理论上的 Hard-DGM 和实际 top-(k) centroid retrieval 之间仍然有 gap。

---

## 5. Monotone repair 仍然是条件，不是算法

Theorem 5.6 说：如果 repair 在 scoring buffer 上 monotone，则 local objective 不增加；如果 refinement 通过 sample-split guarded gain test，则 penalized objective 下降。

这个 theorem 是对的，但 Algorithm 2 里 repair 仍然是直接执行：

[
M_{j^\star}\leftarrow \operatorname{Repair}(M_{j^\star},h,y),
\quad
c_{j^\star}\leftarrow c_{j^\star}+\eta_c(h-c_{j^\star})
]

这不一定 monotone。尤其 centroid movement 会改变 soft routing，可能让 scoring buffer loss 上升。

建议把 repair 变成 guarded repair：

1. propose a repair candidate (S^{\text{repair}})；
2. compute:

[
R_b(S^{\text{repair}};B^{\text{score}})
\le
R_b(S;B^{\text{score}})
]

3. 若成立，commit repair；
4. 若不成立，减小 centroid step、只更新 memory、不移动 centroid，或者进入 refinement test。

这样 monotone repair 从“theorem assumption”变成“algorithm property”。这是很重要的，因为现在 theorem 依赖的是一个没有被算法 enforce 的条件。

---

## 6. DGM-Embedding 是主方法；Adapter 应继续弱化

当前稿件已经说 DGM-Embedding 是 canonical large-model layer，DGM-Adapter 是 optional residual extension。这个方向正确。

但 DGM-Adapter 仍然占了一定空间，而且 target choices 会让方法显得松散：

[
v_t=P E[x_{t+1}]
]

[
v_t=P(E[x_{t+1}]-\hat e_t)
]

[
v_t=h_t^{\text{teacher}}-h_t
]

DGM-Embedding 的 target 非常自然：

[
v_t=E[x_{t+1}]
]

所以建议正式论文主文只保留 DGM-Embedding，Adapter 放 extension/appendix。否则主方法会从“清楚的 next-token runtime memory”变成“可以有很多 target 的大框架”，这会削弱方法锐度。

---

## 7. 训练时 representation drift 仍然是大问题

DGM 的 anchors、centroids、edges 都存在于 query/hidden space 里。如果 slow parameters (\theta,Q) 在训练过程中变化，那么之前创建的 anchors/centroids/edges 是在旧表示空间中定义的。新表示一变，旧 DGM state 可能 stale。

你现在已经提到 frozen or slowly trained backbone，但还没有给出足够明确的 protocol。

建议明确给出三种合法训练模式：

### Mode A: Frozen encoder

先训练 backbone，然后冻结 (\phi_\theta,Q,E)，只运行 DGM runtime learning。最干净，最适合第一篇实验。

### Mode B: EMA query encoder

DGM 的 query space 使用 EMA encoder：

[
\bar\theta \leftarrow \beta \bar\theta+(1-\beta)\theta
]

anchors/centroids 基于 (\bar\theta)，减少 drift。

### Mode C: periodic state rebuild

每隔 (K) steps，重建或清空 DGM state，以避免 state 与表示空间不一致。

如果不处理这个问题，large-model training 部分会被质疑：你训练 backbone 的同时，DGM memory 绑定的是不断漂移的 hidden geometry。

---

## 8. Related work 现在够“有意识”，但还缺深度比较

引言表格已经是很好的一步，但 related work 小节应更像“地图”，而不是“引用清单”。建议按照 DGM 的三个组成组织：

### A. Refine: concept/category growth

包括 ART、online decision trees、Hoeffding trees、Mondrian forests、LVQ、SOM、Growing Neural Gas、prototype learning。这里说明：这些 work 已经有 category creation/refinement；DGM 的新增点是 graph-generated distinctions + local repair memories + neural runtime layer。

### B. Repair: local runtime state update

包括 passive-aggressive learning、RLS、online ridge、fast weights、linear attention、Projection Memory/CPM。这里说明：这些 work 主要解决 fixed representation 内的 update；DGM 先判断 representation 是否 adequate。

### C. Retrieve/adapt: memory and runtime adaptation

包括 kNN-LM/cache LM、Memorizing Transformer、RETRO/RAG、MoE、TTT/TTA、Memory Networks/NTM/DNC。这里说明：这些 work retrieve/adapt，但一般没有 consequence-forced concept refinement。

这会比表格更有说服力，也能防止 reviewer 说“你只是把若干旧方法拼起来”。

---

## 9. 当前稿件最强的贡献应该怎么表述

不要说 DGM 发明了 prototype、edge、memory、runtime update。它没有。

最强贡献应该写成：

[
\boxed{
\text{DGM separates runtime learning into representation refinement and local state repair.}
}
]

然后紧接：

[
\boxed{
\text{It implements this separation as a graph-generated information structure with edge-gated local memories.}
}
]

这两句话比“we introduce a growing graph of anchored concepts...”更有力量。因为前者是原则，后者是实现。

---

## 10. 当前版本的主要风险

### 风险 1：被认为是 ART / prototype / online tree / cache / MoE 的组合

解决：主动承认组件相似性，把 novelty 定位到 refine-or-repair decomposition。

### 风险 2：DGM-Embedding 被认为只是 cache LM

解决：实验必须比较 cache/kNN-LM，并展示 edge/refine 比 plain cache 强。

### 风险 3：theorem 依赖 monotone repair，但算法不 enforce

解决：加 guarded repair commit rule。

### 风险 4：soft centroid retrieval 可能漏掉 correct hard route

解决：candidate recall assumption + union candidate retrieval。

### 风险 5：representation drift 使 runtime memory stale

解决：frozen encoder / EMA encoder / periodic rebuild protocol。

---

## 11. 实验方向必须重新收紧

现在 computational checks 是有用的 sanity check，但如果要证明 DGM，最好的实验不是普通分类，而是 **hidden-regime online sequence prediction**。

你要构造一个任务，其中：

* 相同表面 context 在不同 hidden regime 下有不同 next-token consequence；
* repair-only memory 会混淆；
* cache/kNN 可以记住局部样本，但缺少 concept distinction；
* DGM 可以通过 surprising consequence 创建新 concept + edges；
* 之后 routing 被 edge-gated 分流。

比较：

* cache LM；
* kNN-LM；
* nearest-centroid memory；
* online decision tree；
* repair-only DGM；
* DGM without edges；
* DGM with edges；
* refine-only；
* refine+repair；
* TTT-style adaptation；
* RLS / fast weights。

核心指标：

[
\text{adaptation lag}
]

[
\text{conflict rate}
]

[
\text{number of concepts}
]

[
\text{negative update rate}
]

[
\text{cross entropy / perplexity}
]

这比 MNIST/CIFAR 更符合 DGM 的主张。

---

# 12. 总体评价

main(75) 已经是一个比较成熟的 draft。它现在有：

* 强主线：refine-or-repair；
* 信息结构理论；
* DGM data structure；
* edge-gated prediction；
* statistically guarded refinement；
* canonical DGM-Embedding layer；
* optional DGM-Adapter；
* CPM/Projection Memory bridge；
* 初步 related work 地图。

但它还需要从“完整框架”进一步变成“可防御论文”。

最重要的四个修改是：

1. **把 monotone repair 变成算法 guard。**
2. **补 candidate recall / retrieval completeness assumption。**
3. **处理 encoder representation drift。**
4. **把 related work 写成真正的 ancestor map，而不是只做表格。**

如果这些完成，DGM 的定位会比较稳：

[
\boxed{
\text{不是新的 kNN，不是新的 tree，不是新的 cache，不是新的 TTT；}
}
]

而是：

[
\boxed{
\text{一个把 runtime learning 拆成 concept refinement 和 local repair 的统一机制。}
}
]

这就是它最有价值的地方。
