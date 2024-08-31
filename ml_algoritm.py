# Data set :


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


df = sns.load_dataset("penguins")
# print(df)

sns.pairplot(df, hue="species")
plt.show()
# print(x)