# GNN-LF/HF

Source Code for WWW2021 : Interpreting and Unifying Graph Neural Networks with An Optimization Framework.

## Resources
[[Paper]](http://shichuan.org/doc/105.pdf) & [[PPT]](http://shichuan.org/doc/105_PPT.pdf) & [[Video(in Chinese)]](https://www.bilibili.com/video/BV1Fh411Q7x7) & [[Video(in English)]](https://www.youtube.com/watch?v=CUkrotAwQVI)

## Environment Settings 
* python == 3.8.5
* torch == 1.7.1
* numpy == 1.19.4  
* scipy == 1.6.0  
* networkx == 2.5
* scikit-learn == 0.24.0 
* pandas == 1.2.0

## Data

* **Cora/Citeseer/Pubmed**: [Semi-Supervised Classifcation with Graph Convolutional Networks.](https://github.com/tkipf/gcn)  
* **ACM**: [Heterogeneous Graph Attention Network.](https://github.com/Jhy1993/HAN)  
* **Wiki-CS**: [Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks.](https://github.com/pmernyei/wiki-cs-dataset)  
* **MS-Academic**: [Predict then Propagate: Graph Neural Networks meet Personalized PageRank.](https://github.com/klicperajo/ppnp)  

## Usage

### Input Parameters
* *(required)* -d/--dataset: name for datasets, i.e., cora/citeseer/pubmed/acm/wiki/ms.
* *(required)* -t: model type, PPNP = 0; GNN-LF = 1; GNN-HF = 2.
* *(required)* -f: propagation form, closed-form = 0; iterative-form = 1.
* -l/-labelrate: training rate, i.e., 20 nodes per class for training, default = 20.
* --niter: times for iteration, default = 10.
* --device: GPU number. 
* --reg_lambda: weight for regularization, default = 5e-3.
* --lr: learning rate, default = 0.01


### Command

* Closed-form GNN-LF:
```
python main.py --dataset=cora -t=1 -f=0 --device=0
```
* Iter-form GNN-LF:
```
python main.py --dataset=cora -t=1 -f=1 --device=0
```

## Cite


## Contact 

If you have any questions, please feel free to contact me with zhumeiqi@bupt.edu.cn 


