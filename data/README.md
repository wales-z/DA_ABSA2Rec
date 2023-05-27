# data preprocess

Put your dataset files in this folder, like: `data/reviews_Automotive_5.json`.

In our experiment, we choose Amazon reviews 5-core "Electornics", "Cell Phones and Accessories", "Video Games", "Automotive", "Musical Instruments".

You need to modify this dict: `dataset_to_rawfile_name` in `preprocess.py` if you uses new datasets. This is a dict with dataset abbreviation as key and raw dataset filename as value.

Then change the RS_TASK_NAME in `preprocess.sh` with a dataset abbreviation and run it.

```bash
sh preprocess.sh
```