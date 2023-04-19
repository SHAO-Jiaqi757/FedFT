- The `server_AServer.py` contains the `Aserver` class, that is the auxiliary server of Phase II. 
- `fedft.py` contains aggregation algorithms in Phase III.



# Non-IID data

There are six datasets located in the `dataset/word_generate` directory. These datasets have **different numbers** of clients, and each client has a unique word associated with it. The unique words across all six datasets are **not shared**, meaning that each dataset has its own set of unique words.

The frequency of each dataset's unique words follows the Zipf distribution, which is a power-law probability distribution that models the occurrence of words in natural languages.

|No. Total client| No. Unique Words|
|----|----|
|2000| 726|
|3500| 1052|
|5000| 1256|
|6500| 1498|
|8000| 1643|
|9500| 1778|


# Experiment on Non-IID

## Intra-cluster Experiments
```
Settings: 20 runs to get the average accuracy
 
 PEM  vs. GTF (w. group_size fitting) vs. XTU (w. GRRX) vs. XTF (FedFT, w. group_size fitting, GRRX) vs. TrieHH

```
### 1. No. Clients 2000
Top 5 frequent words with count: [('smog', 244), ('pianist', 103), ('powder', 70), ('hiccups', 51), ('city', 48)]

| Method | $\varepsilon$ | 0.5  | 1.5  | 2.5  | 3.5  | 4.5  | 5.5  | 6.5  | 7.5  | 8.5  | 9.5  |
|--------|-------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| PEM    | recall      | 0.0 | 0.04| 0.18| 0.23| 0.3 | 0.46| 0.35| 0.51| 0.44| 0.41|
| GTF    | recall      | 0.0 | 0.07| 0.18| 0.25| 0.32| 0.37| 0.42| 0.42| 0.48| 0.41|
| XTU    | recall      | 0.37| 0.6 | 0.83| 0.96| 0.96| 0.96| 0.97| 0.99| 0.98| 0.96|
| XTF    | recall      | 0.66| 0.92| 0.97| 1.0 | 0.99| 1.0 | 1.0 | 1.0 | 1.0 | 1.0|
| TrieHH |recall|  0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 

### 2. No. Clients 3500

Top 5 frequent words with count: [('cantaloupe', 378), ('subset', 204), ('tumbler', 119), ('mixer', 88), ('acknowledgment', 84)]


| Method | $\varepsilon$ | 0.5  | 1.5  | 2.5  | 3.5  | 4.5  | 5.5  | 6.5  | 7.5  | 8.5  | 9.5  |
|--------|-------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| PEM    | recall      | 0.0 | 0.0 | 0.09| 0.19| 0.26| 0.29| 0.32| 0.29| 0.3 | 0.28|
| GTF    | recall      | 0.0 | 0.0 | 0.03| 0.18| 0.27| 0.25| 0.31| 0.28| 0.28| 0.27|
| XTU    | recall      | 0.57| 0.9 | 0.95| 0.99| 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| XTF    | recall      | 0.72| 0.99| 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| TrieHH |recall|  0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |



### 3. No. Clients 5000
Top 5 frequent words with count: [('guitarist', 551), ('sailing', 267), ('mattock', 200), ('meadow', 138), ('invasion', 97)]

| Method | $\varepsilon$ | 0.5  | 1.5  | 2.5  | 3.5  | 4.5  | 5.5  | 6.5  | 7.5  | 8.5  | 9.5  |
|--------|-------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| PEM    | recall      | 0.0 | 0.0 | 0.08| 0.26| 0.3 | 0.32| 0.33| 0.37| 0.33| 0.36|
| GTF    | recall      | 0.0 | 0.0 | 0.01| 0.18| 0.24| 0.3 | 0.33| 0.37| 0.35| 0.36|
| XTU    | recall      | 0.67| 0.89| 0.99| 1.0 | 0.99| 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| XTF    | recall      | 0.87| 0.98| 1.0 | 1.0 | 1.0 | 1.0 | 1
| TrieHH |recall|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.04|0.13|0.19|


### 4. No. Clients 6500
Top 5 frequent words with count: [('stockings', 696), ('system', 327), ('command', 240), ('broom', 192), ('plight', 151)]

| Method | $\varepsilon$ | 0.5 | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 | 7.5 | 8.5 | 9.5 |
|--------|-------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| PEM    | recall      | 0.0 | 0.0 | 0.17| 0.4 | 0.46| 0.54| 0.54| 0.53| 0.54| 0.53|
| GTF    | recall      | 0.0 | 0.01| 0.19| 0.35| 0.48| 0.52| 0.51| 0.49| 0.5 | 0.48|
| XTU    | recall      | 0.74| 0.95| 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| XTF    | recall      | 0.91| 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
|TrieHH |recall| 0.0|0.0|0.0|0.0|0.0|0.05|0.14|0.19|0.2|0.21|



### 5. No. Clients 8000
Top 5 frequent words with count: [('shadow', 842), ('commodity', 431), ('disruption', 322), ('vault', 208), ('farmer', 180)]

| Method | $\varepsilon$ | 0.5 | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 | 7.5 | 8.5 | 9.5 |
|--------|-------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| PEM    | recall      | 0.0 | 0.11| 0.2 | 0.38| 0.46| 0.48| 0.49| 0.47| 0.49| 0.5 |
| GTF    | recall      | 0.0 | 0.11| 0.21| 0.32| 0.35| 0.42| 0.48| 0.45| 0.46| 0.45|
| XTU    | recall      | 0.77| 0.92| 0.98| 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| XTF    | recall      | 0.99| 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
|TrieHH|recall| 0.0|0.0|0.0|0.0|0.12|0.19|0.18|0.2|0.24|0.23|



### 6. No. Clients 9500
Top 5 frequent words with count: [('lad', 1023), ('brake', 524), ('tabby', 343), ('manager', 261), ('spending', 208)]


| Method | $\varepsilon$ | 0.5 | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 | 7.5 | 8.5 | 9.5 |
|--------|-------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| PEM    | recall      | 0.0 | 0.22| 0.48| 0.61| 0.68| 0.66| 0.72| 0.66| 0.69| 0.71|
| GTF    | recall      | 0.0 | 0.24| 0.45| 0.6 | 0.66| 0.68| 0.69| 0.69| 0.71| 0.67|
| XTU    | recall      | 0.86| 0.97| 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| XTF    | recall      | 0.96| 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
|TrieHH |recall| 0.0|0.0|0.06|0.17|0.35|0.47|0.61|0.8|0.88|0.91|


## Intra-cluster Experiments

Combine all the data from the 6 clusters and run the experiments on the combined data. 



Combined (34500 clients): 
Top 5 frequent words with count: [('lad', cluster 6), ('shadow', cluster 5), ('stockings', cluster 4), ('guitarist', cluster 3), ('brake', cluster 6)]

|$\varepsilon$| 0.5| 1.5| 2.5| 3.5| 4.5| 5.5| 6.5| 7.5| 8.5| 9.5|
|----|----|----|----|----|----|----|----|----|----|----|
|F1|0.0|0.0|0.1|0.36|0.488|0.677|0.763|0.866|0.875|0.942|
|recall| 0.0|0.0|0.06|0.17|0.35|0.47|0.61|0.8|0.88|0.91|
