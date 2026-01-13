#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import operator 


# In[2]:


# Read Intermediate data from the folder
# Change the file name
#df = pd.read_csv("intermediate_data/Data_1.CSV",skiprows=0)


df = pd.read_csv(r'/Users/Stephen/Desktop/Learning_code/name_matching_tool_dealogic/intermediate_data/Data_sample_APAC_dealogic.CSV', encoding= 'unicode_escape')



# In[3]:


###df.columns


# In[4]:


###df.dropna(subset=['Processed Name'], inplace=True) 
#Drop the ones where company_name is missing (Actually it's for removing the NaNs when the csv file is improperly read)
###df


# ## State: 
# ## -1 No match from Capital IQ 
# ### -- Capital IQ search returns no result
# ## 0 No Match
# ### -- Name_matcher program returns no result 
# ## 1 Exact Match (Insensitive to Capital Letters)
# ### -- The Names, after removing blanks, non-ascii chars, ", . etc.", are EXACTLY the same && Their countries are the same
# ### -- Parent Names are EXACTLY the same (For those initially == -1 or 0 or 4) 
# ## 2 Multiple Exact Match
# ### -- Same critetia as 1(NAME+COUNTRY), but there are mutiple EXACT matches, we return the 1st result as default
# ### -- Parent Names are EXACTLY the same and mutiple EXACT (For those initially == -1 or 0)
# ## 3 Very Close Match 
# ### -- Possible outcome 1: Discounted Levenshetein score OR SSK > 0.95 & Country is the same
# ### -- Possible outcome 2: Discounted Levenshetein OR SSK (String Subsequence Kernel Similarity) > 0.75 & Country & Industry is the same
# ### -- Possible outcome 3: Parent outcome for ANY string_score>0.75 (By experience, as Parents are more famous, Capital IQ tends to return better results) (State -1,0,4) will undergo this parent check
# 
# ## 4 Close Match (Unsure). 
# ### -- ONE OF String_match > 0.75 but country or Industry is not the same (Double Checks needed)
# 
# ## 5 Special Match (Special BUT quite sure)
# ### -- Possible outcome 1////: string_match(one of them) < 0.75(Lower) BUT Country1==Country and Sector1 == Sector (The first result satisfying this requirement is usually a renamed company or acquisition etc. as Capital IQ tends to do that, by experience). or simply, it's a good match, with one of the string_match >0.75
# 
# 

# ## Dealing with those having at least 1 match from Capital IQ (State != -1)
# ### State 1,2: Results are SOLID, as even suffixes like LLC, Ltd etc. are exactly the same

# In[5]:


#Deal with the ones where there is at least 1 match
df_match = df.loc[df['MATCH_NAME'] != '(Invalid Identifier)']
df_nomatch = df.loc[df['MATCH_NAME'] == '(Invalid Identifier)']
df_nomatch['state'] = -1 #NoMatch: State = -1

print(len(df_match))


# In[6]:


# Function to removie non-ascii chars
import unicodedata

def remove_non_ascii(text):
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode()


# In[7]:


def exact_match(row):
    check = 0
    #Define lists for storage of results
    matched = []
    matched_id = []
    matched_no = []
    # loop with the FIVE matches
    for i in ['1','2','3','4','5']:
        #Remove special symbols, blanks and make all to be lower-case   
        pro_name = row['Processed Name'].lower().replace(",", "").replace(".", "").replace(" ", "").replace("?","")
        pro_name = remove_non_ascii(pro_name) # remove non-ascii
        match_name = str(row[('MATCH_NAME'+i)]).lower().replace(",", "").replace(".", "").replace(" ","").replace("?","") 
        match_name = remove_non_ascii(match_name)
        #print(pro_name)
        #print(match_name)
        if (pro_name == match_name and  row['country'] == row[('COUNTRY'+i)]):#The country needs to be correct
            matched.append(row[('MATCH_NAME'+i)]) #NAME Match
            matched_id.append(row[('MATCH_ID'+i)]) #ID Match
            matched_no.append(i) #Match_Number
            check = check + 1
    if check == 0: # No match
            return pd.Series([np.nan, np.nan,0,'0'])
    elif check == 1: # Exactly One Match
            #print(matched_id[0])
            return pd.Series([matched[0],matched_id[0],1,matched_no[0]])
    elif check > 1: # More than one match
            #print('Mutiple Matches:'+row['Processed Name']+' Matched:'+str(matched))
            return pd.Series([matched[0],matched_id[0],2,matched_no[0]]) 
            #Using the default first one as the result for now (CAPITAL IQ SHOULD RETURN THE BEST ANSWER FIRST)
            
### WE COULD ADD A COLUMN INDICATING THERE'S MULTIPLE MATCHES (WITH COUNTRY CORRECT), MAYBE NEED TO SEE WHICH ONE CONTAINS MORE INFO
### OR COMPARE THE INDUSTRIES?


# In[8]:


df_match[['result_name','result_id','state','matched_no']] = df_match.apply(exact_match,axis=1)
df_exact_match = df_match[df_match['state']>0] # Store into df_exact_match for combining them later
len(df_match[df_match['state']>0])


# ## Dealing with those that have no EXACT match (State > 2)
# ### We now disregard suffixes like LLC,Corp etc. for better String Similarity Results

# In[9]:


df_non_exact = df_match.loc[df_match['result_id'].isna()] #Those with at least one match, but no exact result


# In[10]:


from cleanco import basename
import re
import abydos.distance as abd

def fuzzy_match(row):
    check = 0
    pre_score = 0
    matched =''
    matched_id = ''
    matched_no = ''
    for i in ['1','2','3','4','5']:
        #Delete suffixes "LLD,Corp etc.",'. ,' using cleanco pack
        pro_name = basename(row['Processed Name'].lower().replace(',','').replace('.','').replace('?',''))
        match_name = basename(str(row[('MATCH_NAME'+i)]).lower().replace(",", "").replace(".", "").replace('?',''))
        pro_name = remove_non_ascii(pro_name) # Remove non-ASCII chars
        match_name = remove_non_ascii(match_name)
        
        if(match_name!='nan'):
            if ((abd.DiscountedLevenshtein().sim(pro_name,match_name)>0.95 or abd.SSK().sim(pro_name,match_name)>0.95) 
                and (row['country'] == row[('COUNTRY'+i)])): 
                #From experience, 0.95 indicates ALMOST exactly the same, only missed by a "s" or some minor differences.
                return pd.Series([row[('MATCH_NAME'+i)],row[('MATCH_ID'+i)],3,i])
            ### stephen's version 2, manually change this threshold from 0.75 to 0.6
            elif (abd.SSK().sim(pro_name,match_name)>0.6 or abd.DiscountedLevenshtein().sim(pro_name,match_name)>0.6):
                #If any of them is larger than 0.75
                if(abd.SSK().sim(pro_name,match_name)>pre_score):
                    #if the new one is larger than the original score
                        matched = row[('MATCH_NAME'+i)] #Checking Names
                        matched_id = row[('MATCH_ID'+i)]
                        check = 1
                        pre_score = abd.SSK().sim(pro_name,match_name)
                        matched_no = i
                        #Checking Industry Sector and Country
                        if(row['IQ_INDUSTRY_SECTOR'] == row[('SECTOR'+i)] and row['country'] == row[('COUNTRY'+i)]):
                            check = 2
                            matched_no = i
                            break
                else:
                    continue
        
    if check == 1:
        return pd.Series([matched,matched_id,4,matched_no]) 
    elif check == 2:
        return pd.Series([matched,matched_id,3,matched_no]) 
    else:
        # Do Industry Check for the 1st Match
        if(row['country'] == row['COUNTRY1'] and row['IQ_INDUSTRY_SECTOR'] == row['SECTOR1']):
            return pd.Series([row[('MATCH_NAME1')],row[('MATCH_ID1')],5,'1']) 
        #Even though names are different, but Capital IQ recognizes those acquisitions or renamed
        return pd.Series([np.nan, np.nan,0,'0'])
                  
#print('Initial: '+pro_name+'  Match: '+match_name + '  SSK Score:'+str(abd.SSK().sim(pro_name,match_name))
#+' Discount Score '+str(abd.DiscountedLevenshtein().sim(pro_name,match_name)))
        


# In[11]:


df_non_exact[['result_name','result_id','state','matched_no']] = df_non_exact.apply(fuzzy_match,axis=1)


# In[12]:


df_semifinal = pd.concat([df_exact_match, df_non_exact,df_nomatch], axis=0) #Combine the above segments of data for Parent Match Round
df_semifinal.sort_index(inplace=True)


# In[13]:


def par_exact_match(row):
    check = 0
    matched = []
    matched_id = []
    par_matched_no = []
    
    if (row['PAR_MATCH_NAME'] == 'NAME') and (row['state'] == 0 or row['state'] == -1 or row['state'] == 4):
        #print(row['PAR_MATCH_NAME'])
        for i in ['1','2','3','4','5']:
            #Remove , . blanks and make all lower case   
            pro_name = row['processed_parent'].lower().replace(",", "").replace(".", "").replace(" ", "").replace("?","")
            pro_name = remove_non_ascii(pro_name)
            match_name = str(row[('PAR_MATCH_NAME'+i)]).lower().replace(",", "").replace(".", "").replace(" ","").replace("?","") 
            match_name = remove_non_ascii(match_name)
            if ( pro_name == match_name):
                # The comparism result
                matched.append(row[('PAR_MATCH_NAME'+i)]) #NAME Match
                matched_id.append(row[('PAR_MATCH_ID'+i)]) #ID Match
                check = check + 1
                par_matched_no.append('p'+i)
            
        if check == 0: # No match
                return pd.Series([np.nan, np.nan,0,'p0'])
        elif check == 1: # Exactly One Match
                #print(matched_id[0])
                return pd.Series([matched[0],matched_id[0],1,par_matched_no[0]])
        elif check > 1: # More than one match
                #print('Mutiple Matches:'+row['processed_parent']+' Matched:'+str(matched)) #Print the ones with mutiple matches
                return pd.Series([matched[0],matched_id[0],2,par_matched_no[0]]) #Using the default first one as the result for now
    else:
        return pd.Series([row['result_name'],row['result_id'],row['state'],row['matched_no']])  #Return the original values
        
        ### WE COULD ADD A COLUMN INDICATING THERE'S MULTIPLE MATCHES (WITH COUNTRY CORRECT), MAYBE NEED TO SEE WHICH ONE CONTAINS MORE INFO
        ### OR COMPARE THE INDUSTRIES?
        
    
    


# In[14]:


df_semifinal[['result_name','result_id','state','matched_no']] = df_semifinal.apply(par_exact_match,axis=1)


# In[15]:


from cleanco import basename
import re
import abydos.distance as abd

def par_fuzzy_match(row):
    check = 0
    matched =''
    matched_id = ''
    matched_no = ''
    pre_score = 0
    
    if (row['PAR_MATCH_NAME'] == 'NAME') and (row['state'] == 0 or row['state'] == -1 or row['state'] == 4):   
        for i in ['1','2','3','4','5']:
            #Delete "LLD,Corp etc.",'. ,' using cleanco pack
            pro_name = basename(row['processed_parent'].lower().replace(',','').replace('.','').replace('?',''))
            match_name = basename(str(row[('PAR_MATCH_NAME'+i)]).lower().replace(",", "").replace(".", "").replace('?',''))
            pro_name = remove_non_ascii(pro_name) # Remove non-ASCII chars
            match_name = remove_non_ascii(match_name)
            
            
            #### similar with above, stephen's version 2 relax threshold from 0.75 to 0.6
            if(match_name!='nan'):
                if (abd.DiscountedLevenshtein().sim(pro_name,match_name)>0.6 or abd.SSK().sim(pro_name,match_name)>0.6): 
                    #Allow parent to meet 0.75 only, as it's more famous, and capital IQ tends to return better results
                    return pd.Series([row[('PAR_MATCH_NAME'+i)],row[('PAR_MATCH_ID'+i)],3,('p'+i)])
                else:
                    return pd.Series([row['result_name'],row['result_id'],row['state'],row['matched_no']])  #Return the original value    
    else:
        return pd.Series([row['result_name'],row['result_id'],row['state'],row['matched_no']])  #Return the original values


# In[16]:


df_semifinal[['result_name','result_id','state','matched_no']] = df_semifinal.apply(par_fuzzy_match,axis=1)


# ## Final Result Display


#### additional step   
### further matching for firms that have same name with the other rows, but since different parent name,
### so failed to match, while the other row has the correct parent name, so match successfully
### since they parent's name is different, copy the results other rows.
### for these cases, I use indicator state: 3, p1


## create new column as indicator

df_semifinal.result_name.fillna(value=np.nan, inplace=True)

## create new column as indicator so we know which row that uses company matched name from another result
df_semifinal["result_name_holding"]= df_semifinal["result_name"]


df_semifinal["result_name"]= df_semifinal.replace("None",np.nan).groupby("Processed Name")["result_name"].transform("first")


df_semifinal.result_name.fillna(value=np.nan, inplace=True)

df_semifinal["result_id"]= df_semifinal.replace("None",np.nan).groupby("Processed Name")["result_id"].transform("first")

df_semifinal.result_id.fillna(value=np.nan, inplace=True)


#### then create new condition, if result_name_holding != result_name, state 3....


df_semifinal.loc[~pd.isnull(df_semifinal['result_name']) & pd.isnull(df_semifinal["result_name_holding"]), "state"] = 3


df_semifinal.loc[~pd.isnull(df_semifinal['result_name']) & pd.isnull(df_semifinal["result_name_holding"]), "matched_no"] = "p1"


df_semifinal = df_semifinal.drop('result_name_holding', axis=1)




###### further mapping where if same parent company but capitaliq doesn't return result due to
## software problem, then we use the output where there is result (ex. if both parents are same name, the capitaliq
## download tool for some reason only returns the first row with the same parent so other rows are left empty)
## so what we are doing here is to copy and paste from the first row with parent's matched name to all other rows with 
## the same parent's name


## create new column as indicator

df_semifinal.result_name.fillna(value=np.nan, inplace=True)

## create new column as indicator so we know which row that uses company matched name from another result
df_semifinal["result_name_holding"]= df_semifinal["result_name"]


df_semifinal["result_name"]= df_semifinal.replace("None",np.nan).groupby("processed_parent")["result_name"].transform("first")


df_semifinal.result_name.fillna(value=np.nan, inplace=True)

df_semifinal["result_id"]= df_semifinal.replace("None",np.nan).groupby("processed_parent")["result_id"].transform("first")

df_semifinal.result_id.fillna(value=np.nan, inplace=True)


#### then create new condition, if result_name_holding != result_name, state 3....


df_semifinal.loc[~pd.isnull(df_semifinal['result_name']) & pd.isnull(df_semifinal["result_name_holding"]), "state"] = 3


df_semifinal.loc[~pd.isnull(df_semifinal['result_name']) & pd.isnull(df_semifinal["result_name_holding"]), "matched_no"] = "p1"


df_semifinal = df_semifinal.drop('result_name_holding', axis=1)








# In[20]:


result_counts = df_semifinal['state'].value_counts()
state_name = ["No Match From Capital IQ: ","No Match from Name_Matcher Program: ","Exact Match: ","Multiple Exact Matches: ",
              "Very Close Match: ","Close Matches(Double Check needed): ","Special Matches: "]
for i in range(-1,6):
    print(state_name[i+1]+str(result_counts[i]))
    #print(str(result_counts[i]))


# In[18]:


sorted_df = df_semifinal[['state', 'matched_no']].value_counts().sort_values(ascending=True).reset_index(name='count')
sorted_df.sort_values(by=['state','matched_no'])

#sorted_df =  df_semifinal[['matched_no']].value_counts().sort_values(ascending=True).reset_index(name='count')


# ## Output result to excel

# In[21]:


df_semifinal.sort_index().to_excel(r'/Users/Stephen/Desktop/Learning_code/name_matching_tool_dealogic/end_result/Matched_Data_Final_APAC.xlsx')
#df_semifinal.sort_index().to_excel(r'/Users/Stephen/Desktop/Learning_code/name_matching_tool/end_result/Matched_Data_Finalv2_41banks.xlsx')


# # Post-Processing Checks (Distribution & Sensitive Checks) & Some unfinished codes

# In[ ]:

    


