# Data Download Instructions

All datasets used in ManifoldFlow experiments are **publicly available**.
No raw data is bundled in this repository. Use the scripts below to download them.

---

## WikiText-2 (Language Modeling)

Used in: LSTM and Mini-Transformer experiments.

```python
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
# train/validation/test splits available
```

Or via manual download:
```bash
pip install datasets
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-2-raw-v1')"
```

---

## Adult Census Income (Tabular Classification)

Used in: MLP experiments (Adult dataset).

```python
from sklearn.datasets import fetch_openml
adult = fetch_openml("adult", version=2, as_frame=True)
```

Or download from UCI:
- URL: https://archive.ics.uci.edu/dataset/2/adult

---

## Covertype (Tabular Classification)

Used in: MLP experiments (Covertype dataset).

```python
from sklearn.datasets import fetch_covtype
covtype = fetch_covtype(as_frame=True)
```

Or download from UCI:
- URL: https://archive.ics.uci.edu/dataset/31/covertype

---

## CIFAR-10 / CIFAR-100 (Image Classification)

Used in: ResNet-18 and LeNet experiments.

```python
import torchvision
torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True)
torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True)
```

Or via HuggingFace:
```python
from datasets import load_dataset
ds10 = load_dataset("uoft-cs/cifar10")
ds100 = load_dataset("uoft-cs/cifar100")
```

---

## Penn Treebank (PTB) (Language Modeling)

Used in: LSTM PTB experiments.

```python
from datasets import load_dataset
ds = load_dataset("ptb_text_only")
```

---

## Graph Datasets (Cora, Citeseer, PubMed, ogbn-arxiv)

Used in: GCN experiments.

```python
from torch_geometric.datasets import Planetoid
cora = Planetoid(root='./data/cora', name='Cora')
citeseer = Planetoid(root='./data/citeseer', name='CiteSeer')
pubmed = Planetoid(root='./data/pubmed', name='PubMed')
```

ogbn-arxiv:
```python
from ogb.nodeproppred import PygNodePropPredDataset
dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/')
```
