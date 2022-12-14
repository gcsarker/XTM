# XTM
<p> This Repository is accompanied by the paper titled, </p>

> ***XTM : A Novel NLP based Transformer & LSTM model to
Detect & Localization of Formally Verified FDI Attack In Smart Grid***. 

<p> False data injection (FDI) attack in smart grid can cause catastrophic impact in energy management and distribution. Here, a novel hybrid model combining the state of the art transformer and LSTM is developed to detect the presence of FDI as well as the location of attack in smart grid. </p>


## Dataset:
***
<p> The initial dataset consists of hourly historical sensor measurements of one year. we have tested our model on <b>IEEE-14 bus system </b>. So, So, we have taken 54 measurements. We have generated the attack vector in accordance with <a href = "https://ieeexplore.ieee.org/abstract/document/9705034/">this article </a>. Further we have extended the hourly dataset to minutely dataset. Both hourly, minutely data and the generated attack vector can be accessed from <a href = "https://drive.google.com/drive/folders/1Z5m7lIJZFJuL_2wvzQ7hL_pYDETxy9uN?usp=sharing"> google drive </a>. </p>

## Setting Up
***
The model is developed on the following system environments,
- python 3.8
- tensorflow 2.7.

Use the following steps to run the model. In the future update the readme file will be modified accordingly.
- clone the repository into the local machine.
- download the data files from drive link above and into the *dataset* directory.
- run *main.py*

## Citation
***


