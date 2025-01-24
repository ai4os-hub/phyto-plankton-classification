

Next, you need add to the [./data/dataset_files](/data/dataset_files) directory the following files:

| *Mandatory files* | *Optional files*  | 
|:-----------------------:|:---------------------:|
|  `classes.txt`, `train.txt` |  `val.txt`, `test.txt`, `info.txt`,`aphia_ids.txt`|

The `train.txt`, `val.txt` and `test.txt` files associate an image name (or relative path) to a label number (that has
to *start at zero*).
The `classes.txt` file translates those label numbers to label names.
The `aphia_ids.txt` file translates those the classes to their corresponding aphia_ids.
Finally the `info.txt` allows you to provide information (like number of images in the database) about each class. 

You can find examples of these files at [./data/demo-dataset_files](/data/demo-dataset_files).

If you don't want to create your own datasplit, this will be done automatically for you with a 80% train, 10% validation, and 10% test split.

