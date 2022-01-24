# Put your dataset files in this folder
In our experiment, we choose Amazon reviews 5-core "Electornics", "Cell Phones and Accessories" and Yelp reviews.

# Dataset Info
Some basic Information for the dataset we use.
## 1 Amazon reviews 5-core
### 1.1 Electronics
user num: 192253
item num: 63001
review/rating num: 1687517 
### 1.2 Cell Phones and Accessories
user num: 27845
item num: 10429
review/rating num: 194204 

> tiny version: (choose former 30000 / 50000 and filter out users with at least 5 interactions)
> user num: 814 / 1484
> item num: 1370 / 1988
> review/rating num: 4909 / 9146 

## 2 Yelp reviews
yelp数据集提供了3个对评论质量进行评价的维度并由用户自行评价（类似对评论点赞、点踩）：userful, funny, cool（以下都是 review 数量）
original: 8635403 
when filtering with useful>0: 3877235
when filtering with funny>0: 1596435
when filtering with cool>0: 1948423
when filtering with useful>0 & cool>0: 1679558
when filtering with all 3 conditions above: 954713

when filtering with 5-core setting: 5766970
when filtering with 10-core setting: 4219140

when filtering with 5-core & useful>0: 2334990
when filtering with 5-core & useful>0 & cool>0: 1005090