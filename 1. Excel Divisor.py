#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import os


# ## Divide

# In[13]:


#df = pd.read_csv(r'/Users/Stephen/Desktop/Learning_code/name_matching_tool/unique_data_4python_step1_8banks.CSV')
df = pd.read_csv(r'/Users/Stephen/Desktop/Learning_code/name_matching_tool_dealogic/unique_data_4python_step1.csv')




# In[14]:


#df.drop(columns=['majorindustrygroup'],axis=1,inplace=True)


# In[17]:


chunk_size = 4000
output_folder = r'/Users/Stephen/Desktop/Learning_code/name_matching_tool_dealogic/raw_data'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

total_rows = len(df)
num_files = (total_rows // chunk_size) + 1

for i in range(num_files):
    start = i * chunk_size
    end = (i + 1) * chunk_size
    chunk_df = df[start:end]

    output_file = os.path.join(output_folder, f'data_section_{i+1}.csv')
    chunk_df.to_csv(output_file, index=False)


# ## Combine

# In[15]:

'''
import os
import pandas as pd

def merge_excel_files(folder_path):
    all_data = pd.DataFrame()
    for file_name in os.listdir(folder_path):
        print(file_name)
        if file_name.endswith('.xlsx'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_excel(file_path)
            all_data = all_data.append(df, ignore_index=True)
    return all_data


# In[16]:


folder_path = 'matched_data/'
merged_data = merge_excel_files(folder_path)


# In[17]:


merged_data


# In[18]:


output_path = 'combined_matched_data.xlsx'
merged_data.to_excel(output_path, index=False)


# In[ ]:




