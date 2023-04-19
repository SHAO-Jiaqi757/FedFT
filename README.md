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

## TrieHH

```
Settings: 20 runs to get the average accuracy


```

2000 clients: 
Top 5 frequent words with count: [('smog', 244), ('pianist', 103), ('powder', 70), ('hiccups', 51), ('city', 48)]

|$\varepsilon$| 0.5| 1.5| 2.5| 3.5| 4.5| 5.5| 6.5| 7.5| 8.5| 9.5|
|----|----|----|----|----|----|----|----|----|----|----|
|F1|  0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|recall|  0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |


3500 clients: 
Top 5 frequent words with count: [('cantaloupe', 378), ('subset', 204), ('tumbler', 119), ('mixer', 88), ('acknowledgment', 84)]


|$\varepsilon$| 0.5| 1.5| 2.5| 3.5| 4.5| 5.5| 6.5| 7.5| 8.5| 9.5|
|----|----|----|----|----|----|----|----|----|----|----|
|F1|  0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|recall|  0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

5000 clients: 
Top 5 frequent words with count: [('guitarist', 551), ('sailing', 267), ('mattock', 200), ('meadow', 138), ('invasion', 97)]

|$\varepsilon$| 0.5| 1.5| 2.5| 3.5| 4.5| 5.5| 6.5| 7.5| 8.5| 9.5|
|----|----|----|----|----|----|----|----|----|----|----|
|F1| 0.0|0.0|0.0|0.0|0.0|0.033|0.083|0.1|0.15|0.233|
|recall|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.04|0.13|0.19|

6500 clients: 
Top 5 frequent words with count: [('stockings', 696), ('system', 327), ('command', 240), ('broom', 192), ('plight', 151)]

|$\varepsilon$| 0.5| 1.5| 2.5| 3.5| 4.5| 5.5| 6.5| 7.5| 8.5| 9.5|
|----|----|----|----|----|----|----|----|----|----|----|
|F1| 0.0|0.0|0.0|0.0|0.017|0.15|0.233|0.283|0.3|0.333|
|recall| 0.0|0.0|0.0|0.0|0.0|0.05|0.14|0.19|0.2|0.21|

8000 clients: 
Top 5 frequent words with count: [('shadow', 842), ('commodity', 431), ('disruption', 322), ('vault', 208), ('farmer', 180)]

|$\varepsilon$| 0.5| 1.5| 2.5| 3.5| 4.5| 5.5| 6.5| 7.5| 8.5| 9.5|
|----|----|----|----|----|----|----|----|----|----|----|
|F1| 0.0|0.0|0.0|0.1|0.217|0.317|0.333|0.333|0.345|0.345|
|recall| 0.0|0.0|0.0|0.0|0.12|0.19|0.18|0.2|0.24|0.23|

9500 clients: 
Top 5 frequent words with count: [('lad', 1023), ('brake', 524), ('tabby', 343), ('manager', 261), ('spending', 208)]

|$\varepsilon$| 0.5| 1.5| 2.5| 3.5| 4.5| 5.5| 6.5| 7.5| 8.5| 9.5|
|----|----|----|----|----|----|----|----|----|----|----|
|F1|0.0|0.0|0.05|0.25|0.333|0.345|0.417|0.405|0.452|0.545|
|recall|  0.0|0.0|0.01|0.14|0.2|0.2|0.25|0.28|0.34|0.38|


Combined (34500 clients): 
Top 5 frequent words with count: [('lad', 1024), ('shadow', 850), ('stockings', 697), ('guitarist', 554), ('brake', 527)]

|$\varepsilon$| 0.5| 1.5| 2.5| 3.5| 4.5| 5.5| 6.5| 7.5| 8.5| 9.5|
|----|----|----|----|----|----|----|----|----|----|----|
|F1|0.0|0.0|0.1|0.36|0.488|0.677|0.763|0.866|0.875|0.942|
|recall| 0.0|0.0|0.06|0.17|0.35|0.47|0.61|0.8|0.88|0.91|
