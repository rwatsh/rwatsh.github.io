---
layout: post
title: Setting up Google Colab Environment with Google Drive and Github
date: 2024-05-05 01:24 -0500
tags: [ML, ds]
categories: [DS, ML]
---
- Colab provides TPUs and we can use the environment to run our code in Colab VM. Running ML code locally is slow.
- Mount google drive in Colab Notebook with following lines of code.

````bash
from google.colab import drive
drive.mount('/content/drive')
````

- Then use the [Google drive desktop](https://www.google.com/drive/download/) to install the Google drive on local and it will be mounted to local filesystem.
- From there we can use the github code repo and push the changes done in Google Colab in GitHub repo.
- Once the drive is mounted inside Colab VM, we can also reference any files in the Google drive as shown below.

```python
import pandas as pd
data = pd.read_csv("/content/drive/My Drive/myProject/datasets/iris.csv")
data.head()
```

