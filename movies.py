#!/usr/bin/env python
# coding: utf-8

# <h3>Project: Investigate a Dataset (TMDb_Movies Dataset)</h3>
# 
# This data set contains information about 10,000 movies collected from The Movie Database (TMDb), including user ratings, budget, cast and revenue.
#   
# For the dataset, i would like to pose the following questions
# <b><h2>Questions</h2></b>
# <ol>
#     <li>Year in which most movies were released</li>
#     <li>Popular Genre</li>
#     <li>Movie Genre by Highest/Most Vote Counts</li>
#     <li>Popular Keywords</li>
#     <li>Top-ten movies by revenue</li>
#     <li>Do Popularity depend on Runtime</li>
#     <li>Movie Runtime over the years</li>
#     <li>Popularity of Big Budget Movies</li>
#     <li>Popular Actors</li>
# </ol>

# <h3>Importing Data</h3>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

movie = pd.read_csv('tmdb-movies.csv')
movie.head()


# <h3>Cleaning Data</h3>

# In[2]:


movie.shape


# In[3]:


movie.info()


# In[4]:


#drop columns not needed for analysis
movie.drop(['id', 'imdb_id', 'homepage', 'tagline', 'overview', 'budget_adj', 'revenue_adj'], axis=1, inplace=True)


# In[5]:


movie.info()


# In[6]:


movie.describe()


# In[7]:


movie.hist(figsize=(20,20));


# In[8]:


#split genre, cast, production_companies & keywords
movie['genre'] = movie['genres'].str.split('|', expand=True)[0]
movie['genres'] = movie['genres'].str.replace('|',',')

movie['actor'] = movie['cast'].str.split('|', expand=True)[0]
movie['actor'] = movie['cast'].str.replace('|',',')

movie['keyword'] = movie['keywords'].str.split('|', expand=True)[0]
movie['keyword'] = movie['keywords'].str.replace('|',',')

movie['production_companies'] = movie['production_companies'].str.split('|', expand=True)[0]


# In[9]:


# convert budget & revenue value to billion
movie['budget'] = movie['budget']/100000000
movie['revenue'] = movie['revenue']/100000000


# In[10]:


# rename budget & revenue columns
movie = movie.rename(columns={'budget': 'budget_in_billion', 'revenue': 'revenue_in_billion'})


# In[11]:


movie.head()


# <h3>Exploration Phase</h3>

# <b><h3>Year in which most movies were released.</h3></b>

# In[12]:


plt.figure(figsize=(25, 15))
sns.countplot(x='release_year', data=movie)
plt.show()


# <i>This visualisation aims to find out in which year most movies where released. From the figure, we can conclude that <b>2014</b> saw most movie releases, followed by <b>2013 & 2015.</b></i>

# <b><h3>Popular Genres</h3></b>

# In[13]:


# mean of popularity for all the genres
popular_genre = movie.groupby('genre')['popularity'].mean().sort_values(ascending=False)
popular_genre


# <i>From the analysis <b>Adventure</b> is most popular genre, followed by <b>Sci-fi, Fantasy & Action.</b></i>

# <b><h3>Movie Genre by Highest/Most Vote Counts</h3></b>

# In[14]:


plt.figure(figsize=(25, 15))
sns.barplot(x='genre', y='vote_count', data=movie)
plt.show()


# <i>This visualization, shows a barplot of Genre vs Vote Count, to determine which genre has the highest count. From the analysis it is clear that, <b>Adventure</b> genre has the most number of counts followed by <b>Sci-fi, Action & Fantasy.<br> The finding also support our previous analysis of popular genres.</b></i>

# <b><h3>Popular Keywords</h3></b>

# In[15]:


# split the keywords
split_keyword = movie['keyword'].str.split(',', expand=True)[0]
#store in a variable
key_word =  movie.groupby('keyword')['popularity'].count().sort_values(ascending=False)
# top-10 popular keywords
popular_keywords = key_word.head(10)


# In[16]:


plt.figure(figsize=(15,15))
plt.ylabel('Popularity')
popular_keywords.plot(kind='bar', color='g', alpha=0.5)
plt.show()


# <p><i>From the above analysis we can conclude that the popular keywords people use to search a movie are <b>woman director</b> followed by <b>independent film, sport, musical, suspense. Those keywords can be helpful while building a movie recommendation system.</b></i></p>

# <b><h3>Top-ten movies by Revenue</h3></b>

# In[17]:


# get the revenue of the movies
top_movies = movie.groupby('original_title')['revenue_in_billion'].sum().sort_values(ascending=False)
top_ten = top_movies.head(10)
top_ten


# In[18]:


plt.figure(figsize=(20,20))
plt.ylabel('Revenue in Billion')
top_ten.plot(kind='bar', color='r', alpha=0.5)
plt.show()


# <i>This visualisation, shows the analysis of the revenues of the top-ten movies according to the revenue they generated. From the figure above we can conclude that <b>Avatar</b> has the highest revenue, closely followed by <b>Star Wars: The force awakens & Titanic</b></i>

# <h3>Do Popularity depend on Runtime</h3>

# In[19]:


plt.figure(figsize=(20,10))
sns.relplot(x='runtime', y='popularity', size="runtime", data=movie)
plt.show()


# <i>In this analysis, we want to see if the runtime of the movie determine the popularity of movie. From the figure above we can see that movies with runtime between <b> 100 & 150 </b> are most popular.</i>

# <h3>Movie Runtime over the years</h3>

# In[20]:


# get the avg.runtime of movies by the year they were released
avg_runtime = movie.groupby('release_year')['runtime'].mean()

plt.figure(figsize=(10, 10))
plt.xlabel('Years')
plt.ylabel('Runtime')
plt.plot(avg_runtime)
plt.show()


# <i>The plot shows the average runtime of movies over the years. As we can see the runtime of movies have been reducing, movies released in the <b>1960's had runtime of around 125 minutes.</b> Although there was an increase in runtime in the mid <b>1980's</b>, movies <b>post 2007 have a runtime of less than 100 minutes.</b></i>

# <h3>Popularity of Big Budget Movies</h3>

# In[21]:


# sort movies according to popularity
most_popular = movie.sort_values(by=['popularity'], ascending=False).head()
most_popular


# In[22]:


# get budget for those movies
mp_budget = most_popular.groupby('original_title')['budget_in_billion'].sum()


# In[23]:


plt.figure(figsize=(10,10))
plt.ylabel('Budget in Billion')
mp_budget.plot(kind='bar', color='#ff5733', alpha=0.5)
plt.show()


# <i>From the above analysis we can conclude that big budget movies tend to be most popular among audience</i>

# <h3>Popular Actors</h3>

# In[24]:


actor = movie['actor'].str.split(',', expand=True)[0]
actor.value_counts().sort_values(ascending=False).head(10)


# <i>From the above analysis we can conclude that popular actors are <b>Nicolas Cage</b> followed by <b>Robert De Niro & Bruce Willis.</b></i>

# <h3>Conclusion</h3>

# From all the analysis made on the Movies and their Genre, Runtime, Popularity, Budget, Actors. The conclusion can be made as-
# 
# Movies tend make revenues based on: Genre, Runtime, Popularity, Actors(lead role)
# 
# So from the given data, movies with Genre like Adventure,Science Fiction,Action and Fantasy, based on the popularity and the run time, the actors in the movies; a movie can make a great ROI.
