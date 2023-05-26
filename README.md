- The `server_AServer.py` contains the `Aserver` class, that is the auxiliary server of Phase II. 
- `fedft.py` contains aggregation algorithms in Phase III.



# Experiment on Non-IID

There are six datasets located in the `dataset/word_generate` directory. These datasets have **different numbers** of clients, and each client has a unique word associated with it. The unique words across all six datasets are **not shared**, meaning that each dataset has its own set of unique words.

The frequency of each dataset's unique words follows the Zipf distribution, which is a power-law probability distribution that models the occurrence of words in natural languages.


Non-IID -- heterogeneous and unbalanced data: different clusters have different word sets and frequencies, different clusters have different numbers of clients.

|No. Total client| No. Unique Words|
|----|----|
|2000| 726|
|3500| 1052|
|5000| 1256|
|6500| 1498|
|8000| 1643|
|9500| 1778|


## Intra-cluster Experiments
We compared the performance of five methods: PEM, GTF, XTU, XTF (FedFT), and TrieHH. 

Ablation experiment:

| Model | GRR/$GRR_X$ | Uniform/Incremental client-size |
| --- | --- | --- |
| PEM (GTU) | GRR | Uniform |
| GTF | GRR | Incremental |
| XTU | $GRR_X$ | Uniform |
| FedFT (XTF) | $GRR_X$ | Incremental |

```
Settings: 20 runs to get the average accuracy
 
 PEM  vs. GTF (w. group_size fitting) vs. XTU (w. GRRX) vs. XTF (FedFT, w. group_size fitting, GRRX) vs. TrieHH

```
### 1. No. Clients 2000
Top 5 frequent words with count: [('smog', 244), ('pianist', 103), ('powder', 70), ('hiccups', 51), ('city', 48)]

| Method   | varepsilon   |   0.5 |     1.5 |      2.5 |      3.5 |      4.5 |      5.5 |      6.5 |      7.5 |      8.5 |      9.5 |
|:---------|:-------------|------:|--------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| TrieHH   | recall       |     0 | 0       | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        |
| TrieHH   | F1           |     0 | 0       | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        |
| PEM      | recall       |     0 | 0.01    | 0.155    | 0.215    | 0.31     | 0.39     | 0.415    | 0.46     | 0.36     | 0.415    |
| GTF      | recall       |     0 | 0       | 0.125    | 0.225    | 0.315    | 0.4      | 0.41     | 0.45     | 0.465    | 0.43     |
| XTU      | recall       |     0 | 0.15    | 0.21     | 0.47     | 0.61     | 0.675    | 0.66     | 0.695    | 0.65     | 0.675    |
| XTF      | recall       |     0 | 0.17    | 0.21     | 0.51     | 0.755    | 0.78     | 0.82     | 0.77     | 0.805    | 0.79     |
| PEM      | F1           |     0 | 0.01625 | 0.140556 | 0.304603 | 0.45873  | 0.524821 | 0.570833 | 0.538988 | 0.545139 | 0.620337 |
| GTF      | F1           |     0 | 0.005   | 0.140198 | 0.23748  | 0.388274 | 0.471528 | 0.528988 | 0.575992 | 0.546429 | 0.611111 |
| XTU      | F1           |     0 | 0.16379 | 0.201111 | 0.46     | 0.565    | 0.685    | 0.68     | 0.695    | 0.66     | 0.69     |
| XTF      | F1           |     0 | 0.165   | 0.2      | 0.53     | 0.755    | 0.8      | 0.805    | 0.775    | 0.81     | 0.77     |
### 2. No. Clients 3500

Top 5 frequent words with count: [('cantaloupe', 378), ('subset', 204), ('tumbler', 119), ('mixer', 88), ('acknowledgment', 84)]


| Method   | varepsilon   |   0.5 |   1.5 |       2.5 |      3.5 |      4.5 |      5.5 |      6.5 |      7.5 |     8.5 |      9.5 |
|:---------|:-------------|------:|------:|----------:|---------:|---------:|---------:|---------:|---------:|--------:|---------:|
| TrieHH   | recall       |     0 | 0     | 0         | 0        | 0        | 0        | 0        | 0        | 0       | 0        |
| TrieHH   | F1           |     0 | 0     | 0         | 0        | 0        | 0        | 0        | 0        | 0       | 0        |
| PEM      | recall       |     0 | 0     | 0.08      | 0.275    | 0.405    | 0.44     | 0.43     | 0.455    | 0.445   | 0.43     |
| GTF      | recall       |     0 | 0     | 0.1       | 0.305    | 0.385    | 0.425    | 0.41     | 0.43     | 0.46    | 0.42     |
| XTU      | recall       |     0 | 0.155 | 0.24      | 0.475    | 0.76     | 0.74     | 0.77     | 0.77     | 0.755   | 0.76     |
| XTF      | recall       |     0 | 0.17  | 0.26      | 0.605    | 0.845    | 0.84     | 0.835    | 0.84     | 0.87    | 0.86     |
| PEM      | F1           |     0 | 0     | 0.0433929 | 0.31996  | 0.420357 | 0.541111 | 0.591667 | 0.621726 | 0.59752 | 0.579762 |
| GTF      | F1           |     0 | 0     | 0.07125   | 0.300754 | 0.398194 | 0.521329 | 0.562718 | 0.612619 | 0.58125 | 0.588393 |
| XTU      | F1           |     0 | 0.09  | 0.23125   | 0.515    | 0.76     | 0.745    | 0.775    | 0.795    | 0.765   | 0.705    |
| XTF      | F1           |     0 | 0.175 | 0.235     | 0.53     | 0.85     | 0.835    | 0.83     | 0.87     | 0.855   | 0.83     |

### 3. No. Clients 5000
Top 5 frequent words with count: [('guitarist', 551), ('sailing', 267), ('mattock', 200), ('meadow', 138), ('invasion', 97)]

| Method   | varepsilon   |   0.5 |   1.5 |      2.5 |      3.5 |      4.5 |      5.5 |      6.5 |     7.5 |      8.5 |      9.5 |
|:---------|:-------------|------:|------:|---------:|---------:|---------:|---------:|---------:|--------:|---------:|---------:|
| TrieHH   | recall       |     0 | 0     | 0        | 0        | 0        | 0        | 0        | 0.04    | 0.13     | 0.19     |
| TrieHH   | F1           |     0 | 0     | 0        | 0        | 0        | 0        | 0.033    | 0.125   | 0.217    | 0.258    |
| PEM      | recall       |     0 | 0     | 0.17     | 0.325    | 0.465    | 0.51     | 0.505    | 0.515   | 0.535    | 0.51     |
| GTF      | recall       |     0 | 0     | 0.15     | 0.315    | 0.45     | 0.51     | 0.535    | 0.515   | 0.52     | 0.51     |
| XTU      | recall       |     0 | 0.18  | 0.235    | 0.56     | 0.74     | 0.825    | 0.78     | 0.79    | 0.75     | 0.795    |
| XTF      | recall       |     0 | 0.19  | 0.225    | 0.595    | 0.865    | 0.88     | 0.855    | 0.845   | 0.845    | 0.84     |
| PEM      | F1           |     0 | 0     | 0.162222 | 0.352976 | 0.45504  | 0.596726 | 0.652798 | 0.64625 | 0.669742 | 0.639107 |
| GTF      | F1           |     0 | 0     | 0.156111 | 0.33625  | 0.460595 | 0.560595 | 0.579306 | 0.64379 | 0.67373  | 0.654067 |
| XTU      | F1           |     0 | 0.18  | 0.2      | 0.565    | 0.76     | 0.79     | 0.8      | 0.81    | 0.735    | 0.78     |
| XTF      | F1           |     0 | 0.195 | 0.22     | 0.605    | 0.835    | 0.855    | 0.86     | 0.82    | 0.85     | 0.835    |


### 4. No. Clients 6500
Top 5 frequent words with count: [('stockings', 696), ('system', 327), ('command', 240), ('broom', 192), ('plight', 151)]

| Method   | varepsilon   |   0.5 |       1.5 |      2.5 |      3.5 |      4.5 |      5.5 |      6.5 |      7.5 |      8.5 |      9.5 |
|:---------|:-------------|------:|----------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| TrieHH   | recall       |     0 | 0         | 0        | 0        | 0        | 0.05     | 0.14     | 0.19     | 0.2      | 0.21     |
| TrieHH   | F1           |     0 | 0         | 0        | 0        | 0.025    | 0.092    | 0.258    | 0.283    | 0.298    | 0.339    |
| PEM      | recall       |     0 | 0         | 0.195    | 0.395    | 0.51     | 0.505    | 0.52     | 0.545    | 0.535    | 0.53     |
| GTF      | recall       |     0 | 0         | 0.205    | 0.4      | 0.47     | 0.53     | 0.515    | 0.525    | 0.515    | 0.55     |
| XTU      | recall       |     0 | 0.18      | 0.23     | 0.58     | 0.845    | 0.87     | 0.805    | 0.85     | 0.885    | 0.85     |
| XTF      | recall       |     0 | 0.195     | 0.25     | 0.62     | 0.905    | 0.905    | 0.9      | 0.87     | 0.94     | 0.91     |
| PEM      | F1           |     0 | 0         | 0.248333 | 0.425556 | 0.514444 | 0.54246  | 0.560238 | 0.58498  | 0.573234 | 0.596508 |
| GTF      | F1           |     0 | 0.0205556 | 0.209643 | 0.365    | 0.47     | 0.546508 | 0.573869 | 0.562024 | 0.599008 | 0.610397 |
| XTU      | F1           |     0 | 0.16      | 0.227361 | 0.565    | 0.83     | 0.84     | 0.87     | 0.85     | 0.845    | 0.815    |
| XTF      | F1           |     0 | 0.19      | 0.22     | 0.645    | 0.93     | 0.92     | 0.905    | 0.93     | 0.915    | 0.91     |


### 5. No. Clients 8000
Top 5 frequent words with count: [('shadow', 842), ('commodity', 431), ('disruption', 322), ('vault', 208), ('farmer', 180)]

| Method   | varepsilon   |   0.5 |   1.5 |   2.5 |   3.5 |      4.5 |      5.5 |      6.5 |      7.5 |      8.5 |      9.5 |
|:---------|:-------------|------:|------:|------:|------:|---------:|---------:|---------:|---------:|---------:|---------:|
| TrieHH   | recall       |     0 | 0     | 0     | 0     | 0.12     | 0.19     | 0.18     | 0.2      | 0.24     | 0.23     |
| TrieHH   | F1           |     0 | 0     | 0     | 0.033 | 0.242    | 0.267    | 0.325    | 0.345    | 0.345    | 0.357    |
| PEM      | recall       |     0 | 0.07  | 0.21  | 0.43  | 0.54     | 0.59     | 0.595    | 0.595    | 0.635    | 0.585    |
| GTF      | recall       |     0 | 0.055 | 0.23  | 0.43  | 0.56     | 0.605    | 0.59     | 0.585    | 0.59     | 0.615    |
| XTU      | recall       |     0 | 0.19  | 0.215 | 0.575 | 0.87     | 0.875    | 0.865    | 0.82     | 0.84     | 0.86     |
| XTF      | recall       |     0 | 0.195 | 0.245 | 0.635 | 0.885    | 0.92     | 0.93     | 0.91     | 0.91     | 0.895    |
| PEM      | F1           |     0 | 0.075 | 0.225 | 0.44  | 0.560556 | 0.660139 | 0.686329 | 0.693294 | 0.714008 | 0.733433 |
| GTF      | F1           |     0 | 0.08  | 0.22  | 0.43  | 0.566111 | 0.61     | 0.665397 | 0.681984 | 0.707738 | 0.720119 |
| XTU      | F1           |     0 | 0.19  | 0.215 | 0.64  | 0.875    | 0.855    | 0.835    | 0.83     | 0.865    | 0.85     |
| XTF      | F1           |     0 | 0.195 | 0.235 | 0.64  | 0.915    | 0.925    | 0.89     | 0.915    | 0.93     | 0.92     |


### 6. No. Clients 9500
Top 5 frequent words with count: [('lad', 1023), ('brake', 524), ('tabby', 343), ('manager', 261), ('spending', 208)]

| Method   | varepsilon   |   0.5 |   1.5 |   2.5 |   3.5 |   4.5 |      5.5 |      6.5 |      7.5 |      8.5 |      9.5 |
|:---------|:-------------|------:|------:|------:|------:|------:|---------:|---------:|---------:|---------:|---------:|
| TrieHH   | recall       | 0     | 0     | 0.06  | 0.17  | 0.35  | 0.47     | 0.61     | 0.8      | 0.88     | 0.91     |
| TrieHH   | F1           | 0     | 0     | 0.067 | 0.283 | 0.325 | 0.333    | 0.387    | 0.44     | 0.47     | 0.51     |
| PEM      | recall       | 0     | 0.195 | 0.39  | 0.55  | 0.65  | 0.67     | 0.64     | 0.635    | 0.655    | 0.64     |
| GTF      | recall       | 0     | 0.185 | 0.39  | 0.56  | 0.64  | 0.67     | 0.685    | 0.65     | 0.67     | 0.67     |
| XTU      | recall       | 0     | 0.2   | 0.36  | 0.595 | 0.87  | 0.88     | 0.83     | 0.835    | 0.825    | 0.845    |
| XTF      | recall       | 0     | 0.2   | 0.43  | 0.635 | 0.925 | 0.925    | 0.885    | 0.94     | 0.905    | 0.91     |
| PEM      | F1           | 0     | 0.2   | 0.39  | 0.55  | 0.63  | 0.635417 | 0.659306 | 0.698889 | 0.674444 | 0.675972 |
| GTF      | F1           | 0     | 0.175 | 0.365 | 0.565 | 0.64  | 0.645    | 0.657222 | 0.664306 | 0.682222 | 0.694722 |
| XTU      | F1           | 0     | 0.19  | 0.395 | 0.63  | 0.84  | 0.855    | 0.865    | 0.83     | 0.855    | 0.835    |
| XTF      | F1           | 0.005 | 0.215 | 0.435 | 0.645 | 0.91  | 0.915    | 0.895    | 0.93     | 0.935    | 0.885    |


## Intra-cluster Experiments

[Cluster 1]: Top 5 frequent words with count: [('smog', 244), ('pianist', 103), ('powder', 70), ('hiccups', 51), ('city', 48)]
[Cluster 2]: Top 5 frequent words with count: [('cantaloupe', 378), ('subset', 204), ('tumbler', 119), ('mixer', 88), ('acknowledgment', 84)]
[Cluster 3]: Top 5 frequent words with count: [('guitarist', 551), ('sailing', 267), ('mattock', 200), ('meadow', 138), ('invasion', 97)]
[Cluster 4]: Top 5 frequent words with count: [('stockings', 696), ('system', 327), ('command', 240), ('broom', 192), ('plight', 151)]
[Cluster 5]: Top 5 frequent words with count: [('shadow', 842), ('commodity', 431), ('disruption', 322), ('vault', 208), ('farmer', 180)]
[Cluster 6]: Top 5 frequent words with count: [('lad', 1023), ('brake', 524), ('tabby', 343), ('manager', 261), ('spending', 208)]


Combined (34500 clients of 6 clusters): 
If we estimate the frequent words from the combined data, we get
Top 5 frequent words : [('lad', cluster 6), ('shadow', cluster 5), ('stockings', cluster 4), ('guitarist', cluster 3), ('brake', cluster 6)]
But frequnt words from cluster 1, 2 are not in the top 5 frequent words.

<!-- 

Combine all the data from the 6 clusters and run the experiments on the combined data. 

|Method|$\varepsilon$| 0.5| 1.5| 2.5| 3.5| 4.5| 5.5| 6.5| 7.5| 8.5| 9.5|
|----|----|----|----|----|----|----|----|----|----|----|----|
|TrieHH|F1|0.0|0.0|0.1|0.36|0.488|0.677|0.763|0.866|0.875|0.942|
|TrieHH|recall| 0.0|0.0|0.06|0.17|0.35|0.47|0.61|0.8|0.88|0.91|

------- -->


Our method estimates the unbiased word frequency in non-IID data by using weighted average relative frequencies. It computes the relative frequency of each word in each cluster, assigns a weight to each cluster proportional to its number of clients, and averages the relative frequencies over all clusters. This method can detect the most frequent words across all clients and within each cluster. 
We apply this method to six clusters with non-IID data, where each cluster has a distinct number of clients and each client has a single word associated with it. 
Each cluster has a unique word set whose frequency follows the Zipf distribution, which is a power-law probability distribution that models the occurrence of words in natural languages.

**Desired** Top 5 frequent words: [('lab', cluster 6), ('shadow', cluster 5), ('stockings', cluster 4), ('guitarist', cluster 3), ('cantaloupe', cluster 2)]


|Method|$\varepsilon$| 0.5| 1.5| 2.5| 3.5| 4.5| 5.5| 6.5| 7.5| 8.5| 9.5|
|----|----|----|----|----|----|----|----|----|----|----|----|
|TrieHH|F1|0.0|0.0|0.117|0.282|0.481|0.65|0.692|0.722|0.761|0.772|
|TrieHH|recall|0.0|0.0|0.035|0.2|0.34|0.455|0.56|0.655|0.72|0.76|
|FedFT|F1|0.00|0.03|0.76|0.79|0.88|0.91|0.93|0.92|0.92|0.96|
|FedFT|recall|0.00|0.03|0.78|0.83|0.90|0.91|0.88|0.93|0.90|0.92|



