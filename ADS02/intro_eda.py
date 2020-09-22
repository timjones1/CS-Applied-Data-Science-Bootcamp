import numpy as np
import pandas as pd
from itertools import chain
def nan_processor(df, replacement_str):
    df.replace(replacement_str, np.nan, inplace=True)
    df.dropna(inplace=True)
   #df = df.apply(pd.to_numeric)
    return df

def feature_cleaner(df, low, high):

   quantile_df = df.quantile([low,high])

   filtered_df = df.apply(lambda x: x[(x > quantile_df.loc[low, x.name]) & (x < quantile_df.loc[high, x.name])], axis=0)
   filtered_df.dropna(inplace=True)
   normalized_df = (filtered_df - filtered_df.mean()) / filtered_df.std()
   return normalized_df

import pandas as pd
import numpy
def get_feature(df):
 #df = pd.DataFrame(data)

 Classvals = df['CLASS']

 max_column = '';
 max_val = 0;
 for (columnName, columnData) in df.iteritems():
        if(columnName != 'CLASS'):
                neg_class = []
                pos_class = []
                for i in range(Classvals.size):
                        if(Classvals[i] == 0):
                                 neg_class.append(columnData[i])
                        else:
                                 pos_class.append(columnData[i])

                # Operations for class 0
                max_0 = numpy.amax(neg_class)
                min_0 = numpy.amin(neg_class)
                var_0 = numpy.var(neg_class)
                max_1 = numpy.amax(pos_class)
                min_1 = numpy.amin(pos_class)
                var_1 = numpy.var(pos_class)

                R_0 = (max_0 - min_0)/var_0
                R_1 = (max_1 - min_1)/var_1

                if(R_0 > R_1):
                        K = R_0/R_1
                else:
                        K = R_1/R_0
                if(K>max_val):
                        max_val = K
                        max_column = columnName

 return max_column






def one_hot_encode(label_to_encode, labels):
   mylist = list(labels)

   b=np.zeros(len(labels))
   if label_to_encode not in mylist:
      return list(int(i)for i in b)
  
   df_dummies = pd.get_dummies(mylist, prefix_sep="__", columns=mylist)
   df_dummies = df_dummies.loc[:, mylist]
   print(df_dummies)

   b= df_dummies.loc[df_dummies.columns==label_to_encode]
   alist=b.values.tolist()

   k=np.fromiter(chain(*alist), dtype=int)
   c=k.tolist()
   return c