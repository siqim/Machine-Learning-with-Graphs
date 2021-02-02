# 1. Introduction
This repo summarizes papers I've read for machine learning on graphs. I'm also writing tutorials on [zhihu.com](https://www.zhihu.com/people/miao-si-qi/columns) and they're in Chinese.

# 2. Requirements
I use basic packages from Anaconda3 with Python 3.8.5. To make my life easier, I also use the following packages to implement models. Please see `requirements.txt` for the full list.
```
torch==1.7.0
torch_geometric==1.6.3
ogb==1.2.3
scikit-multilearn==0.2.0
```
# 3. Papers
The following are papers that I'll cover in this repo. 
## 3.1 Early Research
### 3.1.1 Factorization-Based Models
- **Distributed large-scale natural graph factorization.**
*Amr Ahmed, Nino Shervashidze, Shravan Narayanamurthy, Vanja Josifovski, and Alexander J Smola.*
   WWW 2013.
   
- **Grarep: Learning graph representations with global structural information.**
*Shaosheng Cao, Wei Lu, and Qiongkai Xu.*
   CIKM 2015.
  
- **Asymmetric transitivity preserving graph embedding.**
*Mingdong Ou, Peng Cui, Jian Pei, Ziwei Zhang, and Wenwu Zhu.*
  KDD 2016.

### 3.1.2 Random Walk-Based Models
- **Deepwalk: Online learning of social representations.**
*Bryan Perozzi, Rami Al-Rfou, and Steven Skiena.*
   KDD 2014.

- **node2vec: Scalable feature learning for networks.**
*Aditya Grover and Jure Leskovec.*
   KDD 2014.
  
- **struc2vec: Learning node representations from structural identity.**
*Leonardo FR Ribeiro, Pedro HP Saverese, and Daniel R Figueiredo.*
   KDD 2017.

### 3.1.3 GCN-based Models

## 3.2 Scalability and Expressivity
### 3.2.1 Node Sampling
### 3.2.2 Subgraph Sampling
### 3.2.3 Regularization
### 3.2.4 Architecture

## 3.3 Incorporating Edge and Label Information
### 3.3.1 Incorporating Edge Information
### 3.3.2 Incorporating Label Information

## 3.4 Training Strategy

## 3.5 Generalization to Heterogeneous Graphs
### 3.5.1 Random Walk-Based Models
### 3.5.2 GCN-Based Models

### 3.5.3 Application
## 3.6 Interpretability and Theory Guidance
### 3.6.1 Expressive Power of GCNs
### 3.6.2 When Will GCNs Fail
### 3.6.3 How to Design Better GCNs