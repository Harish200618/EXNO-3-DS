## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="362" height="302" alt="image" src="https://github.com/user-attachments/assets/09767b45-ec5f-4db4-8409-7896724ef286" />

```
 from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
 pm=['Hot','Warm','Cold']
 e1=OrdinalEncoder(categories=[pm])
 e1.fit_transform(df[["ord_2"]])
```
<img width="257" height="191" alt="image" src="https://github.com/user-attachments/assets/5ed5402b-9d37-4c56-9454-b2f07c412b98" />

```
 df['bo2']=e1.fit_transform(df[["ord_2"]])
 df
```
<img width="324" height="307" alt="image" src="https://github.com/user-attachments/assets/bf472512-a996-40e0-a42b-ca69cc494594" />

```
 le=LabelEncoder()
 dfc=df.copy()
 dfc['ord_2']=le.fit_transform(dfc['ord_2'])
 dfc
```
<img width="387" height="309" alt="image" src="https://github.com/user-attachments/assets/0389ee3b-935a-4713-80ed-9bf36b4646ad" />

```
 from sklearn.preprocessing import OneHotEncoder
 ohe=OneHotEncoder(sparse=False)
 df2=df.copy()
 enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
 df2=pd.concat([df2,enc],axis=1)
 df2
```
<img width="514" height="299" alt="image" src="https://github.com/user-attachments/assets/08908af5-9a77-45ab-96b9-357b08a0d4cc" />

```
 pd.get_dummies(df2,columns=["nom_0"])
```
<img width="593" height="305" alt="image" src="https://github.com/user-attachments/assets/e390cf78-da59-4b21-8fe3-32e543282ce9" />

```
 from category_encoders import BinaryEncoder
 df=pd.read_csv("data.csv")
 df
 be=BinaryEncoder()
 nd=be.fit_transform(df['Ord_2'])
 df
 dfb=pd.concat([df,nd],axis=1)
 dfb
```
<img width="827" height="543" alt="image" src="https://github.com/user-attachments/assets/118dd182-fbb9-43fe-993c-f37f7e2ce741" />

```
 from category_encoders import TargetEncoder
 te=TargetEncoder()
 CC=df.copy()
 new=te.fit_transform(X=CC["City"],y=CC["Target"])
 CC=pd.concat([CC,new],axis=1)
 C
```
<img width="680" height="448" alt="image" src="https://github.com/user-attachments/assets/f2372498-e986-479d-8a56-8a6d9d1d6871" />

```
import pandas as pd
 from scipy import stats
 import numpy as np
 df=pd.read_csv("Data_to_Transform.csv")
 df
```
<img width="847" height="528" alt="image" src="https://github.com/user-attachments/assets/552354c1-013e-44ff-8603-7015a5a5ee78" />

```
 df.skew()
```
<img width="433" height="254" alt="image" src="https://github.com/user-attachments/assets/d985d075-15a3-4251-a0ab-a02a216aeb65" />

```
 np.log(df["Highly Positive Skew"])
```
<img width="433" height="546" alt="image" src="https://github.com/user-attachments/assets/f8445bda-3dcb-47c3-a72d-0ab26f8655cd" />

```
 np.reciprocal(df["Moderate Positive Skew"])
```
<img width="499" height="551" alt="image" src="https://github.com/user-attachments/assets/05de1885-b893-4088-8a1e-29097048af91" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="437" height="562" alt="image" src="https://github.com/user-attachments/assets/fead10d6-9da2-49ec-a1bf-d27ade8a3fdb" />

``` 
np.square(df["Highly Positive Skew"])
```
<img width="455" height="543" alt="image" src="https://github.com/user-attachments/assets/7e54e34b-6e60-407d-944f-d8bed6b50948" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="828" height="560" alt="image" src="https://github.com/user-attachments/assets/9af4313e-dde6-4d4d-a51e-39ca9890f062" />

```
 df.skew()
```
<img width="441" height="298" alt="image" src="https://github.com/user-attachments/assets/94fc67d4-9bac-4b26-942c-780b73b2b88a" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
 df.skew()
```
<img width="506" height="320" alt="image" src="https://github.com/user-attachments/assets/19d13119-ca9a-4f7b-814b-d3f672296463" />

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 df
```
<img width="814" height="550" alt="image" src="https://github.com/user-attachments/assets/3e89ffc2-c97a-4ede-a594-275bedcd9555" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="765" height="552" alt="image" src="https://github.com/user-attachments/assets/b1701cf1-bf56-4164-b9d6-10a95cd47ae3" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
<img width="758" height="536" alt="image" src="https://github.com/user-attachments/assets/1b1fb30c-ecb8-46c1-8646-65e8561f91b5" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="760" height="561" alt="image" src="https://github.com/user-attachments/assets/4fce9a64-d0ef-4bfe-8de2-98d6025f40ca" />

```
 df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
 sm.qqplot(df["Highly Negative Skew"],line='45')
 plt.show()
```
<img width="788" height="543" alt="image" src="https://github.com/user-attachments/assets/5502617a-24d6-4789-a0de-c16c84bc701f" />

```
dt=pd.read_csv("/content/Data_to_Transform.csv")
dt
```
```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 dt["Age_1"]=qt.fit_transform(dt[["Age"]])
 sm.qqplot(dt['Age'],line='45') 
plt.show()
```
<img width="691" height="485" alt="image" src="https://github.com/user-attachments/assets/b1ed60b9-dd5f-4ae6-a60d-1e27b0e072a3" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
<img width="771" height="533" alt="image" src="https://github.com/user-attachments/assets/d22ef4d0-ec89-4b93-86a1-7dcea1f99ead" />

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully  

       
