#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import and setup matplotlib:
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # needed for waffle Charts

mpl.style.use('ggplot') # optional: for ggplot-like style

#Import Primary Modules:
import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
from PIL import Image # converting images into arrays

#install seaborn and wordcloud
get_ipython().system('pip install seaborn wordcloud')

#import seaborn
import seaborn as sns

#import wordcloud
import wordcloud

# check for latest version of Matplotlib and seaborn
print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0
print('Seaborn version: ', sns.__version__)
print('WordCloud version: ', wordcloud.__version__)


# In[2]:


#Fetching Data
#Toolkits: The course heavily relies on pandas and Numpy for data wrangling, analysis, and visualization. The primary plotting library we will explore in the course is Matplotlib.

#Dataset: Immigration to Canada from 1980 to 2013 - International migration flows to and from selected countries - The 2015 revision from United Nation's website

#The dataset contains annual data on the flows of international migrants as recorded by the countries of destination. The data presents both inflows and outflows according to the place of birth, citizenship or place of previous / next residence both for foreigners and nationals.

#In this lab, we will focus on the Canadian Immigration data and use the already cleaned dataset.

#You can refer to the lab on data pre-processing wherein this dataset is cleaned for a quick refresh your Panads skill Data pre-processing with Pandas


#Download the Canadian Immigration dataset and read it into a pandas dataframe.


# In[3]:


df_can = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv')

print('Data read into a pandas dataframe!')


# In[4]:


df_can.head()


# In[5]:


# print the dimensions of the dataframe
print(df_can.shape)


# In[6]:


#set Country as index
df_can.set_index('Country', inplace=True)


# In[7]:


#Waffle Charts ¶
#A waffle chart is an interesting visualization that is normally created to display progress toward goals. It is commonly an effective option when you are trying to add interesting visualization features to a visual that consists mainly of cells, such as an Excel dashboard.


# In[8]:


#Let's revisit the previous case study about Denmark, Norway, and Sweden.


# In[9]:


# let's create a new dataframe for these three countries 
df_dsn = df_can.loc[['Denmark', 'Norway', 'Sweden'], :]

# let's take a look at our dataframe
df_dsn


# In[10]:


#Step 1. The first step into creating a waffle chart is determing the proportion of each category with respect to the total.


# In[11]:


# compute the proportion of each category with respect to the total
total_values = df_dsn['Total'].sum()
category_proportions = df_dsn['Total'] / total_values

# print out proportions
pd.DataFrame({"Category Proportion": category_proportions})


# In[12]:


#Step 2. The second step is defining the overall size of the waffle chart.


# In[13]:


width = 40 # width of chart
height = 10 # height of chart

total_num_tiles = width * height # total number of tiles

print(f'Total number of tiles is {total_num_tiles}.')


# In[14]:


#Step 3. The third step is using the proportion of each category to determe it respective number of tiles


# In[15]:


# compute the number of tiles for each category
tiles_per_category = (category_proportions * total_num_tiles).round().astype(int)

# print out number of tiles per category
pd.DataFrame({"Number of tiles": tiles_per_category})


# In[16]:


#Step 4. The fourth step is creating a matrix that resembles the waffle chart and populating it.


# In[17]:


# initialize the waffle chart as an empty matrix
waffle_chart = np.zeros((height, width), dtype = np.uint)

# define indices to loop through waffle chart
category_index = 0
tile_index = 0

# populate the waffle chart
for col in range(width):
    for row in range(height):
        tile_index += 1

        # if the number of tiles populated for the current category is equal to its corresponding allocated tiles...
        if tile_index > sum(tiles_per_category[0:category_index]):
            # ...proceed to the next category
            category_index += 1       
            
        # set the class value to an integer, which increases with class
        waffle_chart[row, col] = category_index
        
print ('Waffle chart populated!')


# In[18]:


waffle_chart


# In[19]:


#Step 5. Map the waffle chart matrix into a visual.


# In[20]:


# instantiate a new figure object
fig = plt.figure()

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()
plt.show()


# In[21]:


#Step 6. Prettify the chart.


# In[22]:


# instantiate a new figure object
fig = plt.figure()

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()

# get the axis
ax = plt.gca()

# set minor ticks
ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
# add gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

plt.xticks([])
plt.yticks([])
plt.show()


# In[23]:


#Step 7. Create a legend and add it to chart.


# In[24]:


# instantiate a new figure object
fig = plt.figure()

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()

# get the axis
ax = plt.gca()

# set minor ticks
ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
# add gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

plt.xticks([])
plt.yticks([])

# compute cumulative sum of individual categories to match color schemes between chart and legend
values_cumsum = np.cumsum(df_dsn['Total'])
total_values = values_cumsum[len(values_cumsum) - 1]

# create legend
legend_handles = []
for i, category in enumerate(df_dsn.index.values):
    label_str = category + ' (' + str(df_dsn['Total'][i]) + ')'
    color_val = colormap(float(values_cumsum[i])/total_values)
    legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

# add legend to chart
plt.legend(handles=legend_handles,
           loc='lower center', 
           ncol=len(df_dsn.index.values),
           bbox_to_anchor=(0., -0.2, 0.95, .1)
          )
plt.show()


# In[25]:


def create_waffle_chart(categories, values, height, width, colormap, value_sign=''):

    # compute the proportion of each category with respect to the total
    total_values = sum(values)
    category_proportions = [(float(value) / total_values) for value in values]

    # compute the total number of tiles
    total_num_tiles = width * height # total number of tiles
    print ('Total number of tiles is', total_num_tiles)
    
    # compute the number of tiles for each catagory
    tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]

    # print out number of tiles per category
    for i, tiles in enumerate(tiles_per_category):
        print (df_dsn.index.values[i] + ': ' + str(tiles))
    
    # initialize the waffle chart as an empty matrix
    waffle_chart = np.zeros((height, width))

    # define indices to loop through waffle chart
    category_index = 0
    tile_index = 0

    # populate the waffle chart
    for col in range(width):
        for row in range(height):
            tile_index += 1

            # if the number of tiles populated for the current category 
            # is equal to its corresponding allocated tiles...
            if tile_index > sum(tiles_per_category[0:category_index]):
                # ...proceed to the next category
                category_index += 1       
            
            # set the class value to an integer, which increases with class
            waffle_chart[row, col] = category_index
    
    # instantiate a new figure object
    fig = plt.figure()

    # use matshow to display the waffle chart
    colormap = plt.cm.coolwarm
    plt.matshow(waffle_chart, cmap=colormap)
    plt.colorbar()

    # get the axis
    ax = plt.gca()

    # set minor ticks
    ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
    ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
    # add dridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    plt.xticks([])
    plt.yticks([])

    # compute cumulative sum of individual categories to match color schemes between chart and legend
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]

    # create legend
    legend_handles = []
    for i, category in enumerate(categories):
        if value_sign == '%':
            label_str = category + ' (' + str(values[i]) + value_sign + ')'
        else:
            label_str = category + ' (' + value_sign + str(values[i]) + ')'
            
        color_val = colormap(float(values_cumsum[i])/total_values)
        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

    # add legend to chart
    plt.legend(
        handles=legend_handles,
        loc='lower center', 
        ncol=len(categories),
        bbox_to_anchor=(0., -0.2, 0.95, .1)
    )
    plt.show()


# In[26]:


#Now to create a waffle chart, all we have to do is call the function create_waffle_chart. Let's define the input parameters:


# In[27]:


width = 40 # width of chart
height = 10 # height of chart

categories = df_dsn.index.values # categories
values = df_dsn['Total'] # correponding values of categories

colormap = plt.cm.coolwarm # color map class


# In[28]:


#And now let's call our function to create a waffle chart.


# In[29]:


create_waffle_chart(categories, values, height, width, colormap)


# In[30]:


#There seems to be a new Python package for generating waffle charts called PyWaffle,

#Let's create the same waffle chart with pywaffle now


# In[33]:


#install pywaffle
get_ipython().system('pip install pywaffle')


# In[34]:


#import Waffle from pywaffle
from pywaffle import Waffle

#Set up the Waffle chart figure

fig = plt.figure(FigureClass = Waffle,
                 rows = 20, columns = 30, #pass the number of rows and columns for the waffle 
                 values = df_dsn['Total'], #pass the data to be used for display
                 cmap_name = 'tab20', #color scheme
                 legend = {'labels': [f"{k} ({v})" for k, v in zip(df_dsn.index.values,df_dsn.Total)],
                            'loc': 'lower left', 'bbox_to_anchor':(0,-0.1),'ncol': 3}
                 #notice the use of list comprehension for creating labels 
                 #from index and total of the dataset
                )

#Display the waffle chart
plt.show()


# In[35]:


#Question: Create a Waffle chart to dispaly the proportiona of China and Inida total immigrant contribution.


# In[38]:


#hint
    #create dataframe
data_CI = .............
    #Set up the Waffle chart figure

fig = plt.figure(FigureClass = ............,
                     rows = ........, columns =....., #pass the number of rows and columns for the waffle 
                     values = ........., #pass the data to be used for display
                     cmap_name = 'tab20', #color scheme
                     legend = {'labels':[.......],
                                'loc': ........., 'bbox_to_anchor':(....),'ncol': 2}
                     #notice the use of list comprehension for creating labels 
                     #from index and total of the dataset
                    )

    #Display the waffle chart
plt.show()


# In[39]:


#Word Clouds 


# In[40]:


#import package and its set of stopwords
from wordcloud import WordCloud, STOPWORDS

print ('Wordcloud imported!')


# In[44]:


#Let's try to analyze a short novel written by Lewis Carroll titled Alice's Adventures in Wonderland. Let's go ahead and download a .txt file of the novel


# In[46]:


import urllib


# In[47]:


# # open the file and read it into a variable alice_novel
alice_novel = urllib.request.urlopen('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/alice_novel.txt').read().decode("utf-8")


# In[48]:


#Next, let's use the stopwords that we imported from word_cloud. We use the function set to remove any redundant stopwords.


# In[50]:


stopwords = set(STOPWORDS)


# In[51]:


#Create a word cloud object and generate a word cloud. For simplicity, let's generate a word cloud using only the first 2000 words in the novel.


# In[52]:


#if you get attribute error while generating worldcloud, upgrade Pillow and numpy using below code
#%pip install --upgrade Pillow 
#%pip install --upgrade numpy


# In[53]:


# instantiate a word cloud object
alice_wc = WordCloud()

# generate the word cloud
alice_wc.generate(alice_novel)


# In[54]:


# display the word cloud
plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[55]:


#Interesting! So in the first 2000 words in the novel, the most common words are Alice, said, little, Queen, and so on. Let's resize the cloud so that we can see the less frequent words a little better.


# In[56]:


fig = plt.figure(figsize=(14, 18))

# display the cloud
plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[57]:


#Much better! However, said isn't really an informative word. So let's add it to our stopwords and re-generate the cloud.


# In[58]:


stopwords.add('said') # add the words said to stopwords

# re-generate the word cloud
alice_wc.generate(alice_novel)

# display the cloud
fig = plt.figure(figsize=(14, 18))

plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[59]:


#Excellent! This looks really interesting! Another cool thing you can implement with the word_cloud package is superimposing the words onto a mask of any shape. Let's use a mask of Alice and her rabbit. We already created the mask for you, so let's go ahead and download it and call it alice_mask.png.


# In[60]:


#save mask to alice_mask
alice_mask = np.array(Image.open(urllib.request.urlopen('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/labs/Module%204/images/alice_mask.png')))


# In[61]:


fig = plt.figure(figsize=(14, 18))

plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[62]:


#Shaping the word cloud according to the mask is straightforward using word_cloud package. For simplicity, we will continue using the first 2000 words in the novel.


# In[63]:


# instantiate a word cloud object
alice_wc = WordCloud(background_color='white', max_words=2000, mask=alice_mask, stopwords=stopwords)

# generate the word cloud
alice_wc.generate(alice_novel)

# display the word cloud
fig = plt.figure(figsize=(14, 18))

plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[64]:


df_can.head()


# In[65]:


#And what was the total immigration from 1980 to 2013?


# In[66]:


total_immigration = df_can['Total'].sum()
total_immigration


# In[67]:


#Using countries with single-word names, let's duplicate each country's name based on how much they contribute to the total immigration.


# In[68]:


max_words = 90
word_string = ''
for country in df_can.index.values:
     # check if country's name is a single-word name
    if country.count(" ") == 0:
        repeat_num_times = int(df_can.loc[country, 'Total'] / total_immigration * max_words)
        word_string = word_string + ((country + ' ') * repeat_num_times)

# display the generated text
word_string


# In[69]:


#We are not dealing with any stopwords here, so there is no need to pass them when creating the word cloud.


# In[70]:


# create the word cloud
wordcloud = WordCloud(background_color='white').generate(word_string)

print('Word cloud created!')


# In[71]:


# display the cloud
plt.figure(figsize=(14, 18))

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[72]:


#Plotting with Seaborn
#In our data 'df_can', let's find out how many continents are mentioned


# In[73]:


df_can['Continent'].unique()


# In[74]:


#countplot
#A count plot can be thought of as a histogram across a categorical, instead of quantitative, variable. Let's find the count of Continents in the data 'df_can' using countplot on 'Continent'


# In[75]:


sns.countplot(x='Continent', data=df_can)


# In[76]:


#The labels on the x-axis doesnot look as expected.
#Let's try to replace the 'Latin America and the Caribbean' with and "L-America", 'Northern America' with "N-America",and change the figure size and then display the plot again


# In[77]:


df_can1 = df_can.replace('Latin America and the Caribbean', 'L-America')
df_can1 = df_can1.replace('Northern America', 'N-America')


# In[78]:


plt.figure(figsize=(15, 10))
sns.countplot(x='Continent', data=df_can1)


# In[79]:


#BarplotThis plot will perform the Groupby on a categorical varaible and plot aggregated values, with confidence intervals.Let's plot the total immigrants Continent-wise


# In[80]:


plt.figure(figsize=(15, 10))
sns.barplot(x='Continent', y='Total', data=df_can1)


# In[81]:


#You can verify the values by performing the groupby on the Total and Continent for mean()


# In[82]:


df_Can2=df_can1.groupby('Continent')['Total'].mean()
df_Can2


# In[83]:


#Regression Plot ¶ With seaborn, generating a regression plot is as simple as calling the regplot function.


# In[84]:


years = list(map(str, range(1980, 2014)))
# we can use the sum() method to get the total population per year
df_tot = pd.DataFrame(df_can[years].sum(axis=0))

# change the years to type float (useful for regression later on)
df_tot.index = map(float, df_tot.index)

# reset the index to put in back in as a column in the df_tot dataframe
df_tot.reset_index(inplace=True)

# rename columns
df_tot.columns = ['year', 'total']

# view the final dataframe
df_tot.head()


# In[85]:


#seaborn is already imported at the start of this lab
sns.regplot(x='year', y='total', data=df_tot)


# In[86]:


#This is not magic; it is seaborn! You can also customize the color of the scatter plot and regression line. Let's change the color to green.


# In[87]:


sns.regplot(x='year', y='total', data=df_tot, color='green')
plt.show()


# In[88]:


#You can always customize the marker shape, so instead of circular markers, let's use +.


# In[89]:


ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+')
plt.show()


# In[90]:


#Let's blow up the plot a little so that it is more appealing to the sight.


# In[91]:


plt.figure(figsize=(15, 10))
sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+')
plt.show()


# In[92]:


#And let's increase the size of markers so they match the new size of the figure, and add a title and x- and y-labels.


# In[93]:


plt.figure(figsize=(15, 10))
ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})

ax.set(xlabel='Year', ylabel='Total Immigration') # add x- and y-labels
ax.set_title('Total Immigration to Canada from 1980 - 2013') # add title
plt.show()


# In[94]:


#And finally increase the font size of the tickmark labels, the title, and the x- and y-labels so they don't feel left out!


# In[95]:


plt.figure(figsize=(15, 10))

sns.set(font_scale=1.5)

ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.show()


# In[96]:


#If you are not a big fan of the purple background, you can easily change the style to a white plain background.


# In[97]:


plt.figure(figsize=(15, 10))

sns.set(font_scale=1.5)
sns.set_style('ticks')  # change background to white background

ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.show()


# In[98]:


#Or to a white background with gridlines.


# In[99]:


plt.figure(figsize=(15, 10))

sns.set(font_scale=1.5)
sns.set_style('whitegrid')

ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.show()


# In[100]:


#Question: Use seaborn to create a scatter plot with a regression line to visualize the total immigration from Denmark, Sweden, and Norway to Canada from 1980 to 2013.


# In[102]:


#The correct answer is:
   
   # create df_countries dataframe
df_countries = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()

   # create df_total by summing across three countries for each year
df_total = pd.DataFrame(df_countries.sum(axis=1))

   # reset index in place
df_total.reset_index(inplace=True)

   # rename columns
df_total.columns = ['year', 'total']

   # change column year from string to int to create scatter plot
df_total['year'] = df_total['year'].astype(int)

   # define figure size
plt.figure(figsize=(15, 10))

   # define background style and font size
sns.set(font_scale=1.5)
sns.set_style('whitegrid')

   # generate plot and add title and axes labels
ax = sns.regplot(x='year', y='total', data=df_total, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigrationn from Denmark, Sweden, and Norway to Canada from 1980 - 2013')


# In[ ]:




