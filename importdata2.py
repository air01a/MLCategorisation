import numpy as np
import pandas as pd
import re

result=pd.DataFrame([])
data = pd.read_csv('data.csv')

for category in data.group.unique():
    if (category!='SUIVI DOUBLONS' and category!='Regul' and category!='IM - LOCAL SUPPORT RUSSIA' and category!='IM - LOCAL SUPPORT HUNGARY'):
        sample =  data[data['group']==category ]
        cnt = sample.count()[0]

        if cnt > 100:
#            print('Adding Category %s' % category)
            number = min(cnt,600)
            result = result.append(sample.sample(cnt))
#        else:
#           print('skipping category %s' % category)

result.to_csv(path_or_buf='./output.csv',index=False,encoding='utf-8')

textfile = open('./output.csv', 'r')
text = textfile.read()
textfile.close()

with open('todelete.csv', 'r') as f:
	tokens = [re.escape(line.strip()) for line in f]

for token in tokens:
	pattern = re.compile('[ \(]'+re.escape(token)+'[ \,\;\:\!\.]', re.IGNORECASE)
	text = pattern.sub(' ', text)

print(text)
