from os import rename
from xml.dom import minicompat
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msn
from sklearn.preprocessing import MinMaxScaler,LabelEncoder

#reduce the size of the properties dataset by selecting smaller datatypes.
'''def reduce_mem_usage(props):

    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.u)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.).min and mx < np.iinfo(np.).max:
                        props[col] = props[col].astype(np.)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist'''

# Load Data
#props_2016 = pd.read_csv('zillow dataset\properties_2016.csv')
#props_2017=pd.read_csv('zillow dataset\properties_2017.csv')

#Run the function
 
#props_2016_1, NAlist = reduce_mem_usage(props_2016)
'''print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)'''

#props_2017_1, NAlist = reduce_mem_usage(props_2017)
'''print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)'''

# the begining of the project

#train_2016=pd.read_csv('zillow dataset/train_2016_v2.csv',parse_dates=["transactiondate"])
#train_2017=pd.read_csv('zillow dataset/train_2017.csv',parse_dates=["transactiondate"])

# check the shape

'''print('the shape of the train_2016 dataset is {}'.format(train_2016.shape))
print('the shape of the train_2017 dataset is {}'.format(train_2017.shape))
print('the shape of the properties_2016 dataset  befor the memory size reduction is {}'.format(props_2016.shape))
print('the shape of the properties_2016 dataset  after the memory size reduction is {}'.format(props_2016_1.shape))
print('the shape of the properties_2017 dataset  befor the memory size reduction is {}'.format(props_2017.shape))
print('the shape of the properties_2017 dataset  after the memory size reduction is {}'.format(props_2017_1.shape))'''

# as expected we didnt lose any data 
# now let's merge the train and properties data 

#merged_2016=pd.merge(props_2016_1,train_2016,on="parcelid",how='left')
#merged_2017=pd.merge(props_2017_1,train_2017,on='parcelid',how='left')
#merged=pd.concat([merged_2016,merged_2017])

#print('the shape of the properties_2016 dataset  after the memory size reduction is {}'.format(merged_2016.shape))
#merged_2017.to_csv('merged_2017.csv')
merged=pd.read_csv('merged_2017.csv')


#First Few Rows Of Data
#print(merged_2016.head(3).transpose())

#Visualizing Datatypes
'''Data_types_2016=pd.DataFrame(merged_2016.dtypes.value_counts().reset_index().rename(columns={'index': 'variable type',0:'cardinal'}))
#Data_types_2017=pd.DataFrame(merged_2017.dtypes.value_counts().reset_index().rename(colomns={'index':'variable type',0:'cardinal'}))
#print(Data_types_2016)

# Visualizing Datatypes 

sns.barplot(x='variable type',y='cardinal',data=Data_types_2016)
plt.show()

#Missing Value Analysis
#visualazation using Barplot
missingValueColumns = merged_2016.columns[merged_2016.isnull().any()].tolist()
data_with_missing_value=merged_2016[missingValueColumns]
msn.bar(data_with_missing_value,figsize=(20,8),color="#34495e",fontsize=12,labels=True)
plt.show()
#visualization using matrix plot
msn.matrix(data_with_missing_value,width_ratios=(10,1), figsize=(20,8),color=(0,0, 0),fontsize=12,sparkline=True,labels=True)
plt.show()
#visualization using the heatmap to check the correlations of nullity between every two colomns 
msn.heatmap(data_with_missing_value,figsize=(20,20))
plt.show()
#visualization using the dendrogram
msn.dendrogram(data_with_missing_value,figsize=(20,20))
plt.show()'''
#Feature Engineering
# Duplicate value removal
#merged_2016.drop_duplicates(inplace=True)
#print('the shape of the properties_2016 dataset  after the memory size reduction is {}'.format(merged_2016.shape))
# as we can see we have no duplicate rows
# Missing value imputation 

#first let's drop the coloumn  transactiondate 
'''merged.drop(['transactiondate'],axis=1,inplace=True) ''' 
#for our data we have the following columns that have missing values
'''List=['hashottuborspa', 'propertycountylandusecode', 'propertyzoningdesc', 'fireplaceflag', 'taxdelinquencyflag', 'logerror', 'transactiondate']
for i in List:
    print('the persentage messimgness in the column {} is {}' .format(i,merged[i].isnull().sum()/merged[i].shape[0]*100))'''
#in our data the follwing rows  'hashottuborspa' ,'fireplaceflag' ,'taxdelinquencyflag','transactiondate'
#  have more than 98 % of messing values so i desided to drop them 
merged.drop(["hashottuborspa", "fireplaceflag",'taxdelinquencyflag','transactiondate','propertycountylandusecode','propertyzoningdesc'], axis = 1, inplace = True)
'''print(merged.shape)
print(merged.head().transpose())'''
#for the  propertycountylandusecode feature we replace the NAN by 

merged['logerror'].fillna(merged['logerror'].mean(),inplace=True)
#Rescaling of incorrectly scaled data
new_merged=pd.get_dummies(data=merged,columns=['heatingorsystemtypeid','propertylandusetypeid','storytypeid','airconditioningtypeid','architecturalstyletypeid','typeconstructiontypeid','buildingclasstypeid'])
print(new_merged.shape)