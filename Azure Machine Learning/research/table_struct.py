import pandas as pd
import numpy as np
from tqdm import tqdm

sines = pd.DataFrame(columns=['angle','sine']) 

for i in tqdm(range(-10, 10)): 
    angle = i / 2.0
    sines = sines.append({'angle':angle,'sine':np.sin(angle)},ignore_index = True)
print(sines[['angle','sine']].to_string(index=False))
