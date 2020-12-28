## 1. Introduction
极大的受到Dimensionality Reduction思想的影响，在图机器学习最早期的研究中涌现了一批基于Matrix Factorization的模型。这类模型往往以重建图的邻接矩阵为目标，学习各个节点的表示。在这篇文章中，我们将介绍其中三篇比较有代表性的工作，包括Graph Factorization<sup>[1]</sup>，GraRep<sup>[2]</sup>以及HOPE<sup>[3]</sup>。同时，在文章最后我们也提供了这三个模型的Pytorch Demo。因笔者能力有限，若有谬误，请不吝指正！

## 2. Graph Factorization
GF<sup>[1]</sup>是Google在13年的一篇文章，可以说是最早的图机器学习模型之一，因此它的思想在今天看来确实十分简单粗暴。

我们直接上目标函数：

$$f(Y, Z, \lambda)=\frac{1}{2} \sum_{(i, j) \in E}\left(Y_{i j}-\left\langle Z_{i}, Z_{j}\right\rangle\right)^{2}+\frac{\lambda}{2} \sum_{i}\left\|Z_{i}\right\|^{2}$$


[1] Amr Ahmed, Nino Shervashidze, Shravan Narayanamurthy,
Vanja Josifovski, and Alexander J Smola. Distributed
large-scale natural graph factorization. In
Proceedings of the 22nd international conference on
World Wide Web, pages 37–48, 2013.

[2] Shaosheng Cao, Wei Lu, and Qiongkai Xu. Grarep:
Learning graph representations with global structural
information. In Proceedings of the 24th ACM international
on conference on information and knowledge
management, pages 891–900, 2015.

[3] Mingdong Ou, Peng Cui, Jian Pei, Ziwei Zhang,
and Wenwu Zhu. Asymmetric transitivity preserving
graph embedding. In Proceedings of the 22nd ACM
SIGKDD international conference on Knowledge discovery
and data mining, pages 1105–1114, 2016.
