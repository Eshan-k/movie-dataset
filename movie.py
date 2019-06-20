import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

movies = pd.read_csv('tmdb-movies.csv')


def print_head():
    print(movies.head(2))


def print_info():
    print(movies.info())

# print_info()
# print_head()


# dropped columns not required for analysis
movies.drop(['homepage', 'tagline', 'keywords', 'overview', 'budget_adj', 'revenue_adj', 'imdb_id'], axis=1, inplace=True)

# split genres, cast & production_companies
movies['genre'] = movies['genres'].str.split('|', expand=True)[0]
movies['genre'] = movies['genres'].str.replace('|', ',')

movies['actors'] = movies['cast'].str.split('|', expand=True)[0]
movies['actors'] = movies['cast'].str.replace('|', ',')

movies['production_companies'] = movies['production_companies'].str.split('|', expand=True)[0]

# fill missing entries
movies['cast'] = movies['cast'].fillna('Unknown')
movies['actors'] = movies['actors'].fillna('Unknown')
movies['director'] = movies['director'].fillna('Unknown')

# convert revenue, budget to million
movies['budget'] = movies['budget']/1000000
movies['revenue'] = movies['revenue']/1000000

# rename the columns
movies = movies.rename(columns={'budget': 'budget_in_million', 'revenue': 'revenue_in_million'})

print_info()
print_head()