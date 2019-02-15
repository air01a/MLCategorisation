import numpy as np
import pandas as pd

result=pd.DataFrame([])
data = pd.read_csv('data.csv')

for category in data.group.unique():
	sample =  data[data['group']==category ]
	cnt = sample.count()[0]

	if cnt > 1:
		print('Adding Category %s' % category)
		number = min(cnt,4000)
		result = result.append(sample.sample(number))
	else:
		print('skipping category %s' % category)

result.to_csv(path_or_buf='./output.csv',index=False,encoding='utf-8')

