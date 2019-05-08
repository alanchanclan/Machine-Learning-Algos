#Dataframe manipulation library
import pandas as pd
#Math functions, we'll only need the sqrt function so let's import only that
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

#Storing the movie information into a pandas dataframe
movies_df = pd.read_csv('movies.csv')
#Storing the user information into a pandas dataframe
ratings_df = pd.read_csv('ratings.csv')

#Head is a function that gets the first N rows of a dataframe. N's default is 5.
print(movies_df.head())

#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
print(movies_df.head())

#Dropping the genres column
movies_df = movies_df.drop('genres', 1)
print(movies_df.head())

# Take a peek at the ratings dataframe
print(ratings_df.head())

#Drop removes a specified row or column from a dataframe, in this case, the timestamp to save memory
ratings_df = ratings_df.drop('timestamp', 1)
print(ratings_df.head())

# Apply it to a movie database with ratings
# Here are the user's preferences
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ]
inputMovies = pd.DataFrame(userInput)
print(inputMovies)

#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('year', 1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original
#dataframe or it might spelled differently, please check capitalisation.
print(inputMovies)

#Filtering out users that have watched movies that the input has watched and storing it
# NOTE: ratings contain the user id's who rated a particular movie, ordered by user ID's in fact
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
print(userSubset.head())

#Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
userSubsetGroup = userSubset.groupby(['userId'])

# Lets look at one of the groups for example (1130)
print(userSubsetGroup.get_group(1130))

#Sorting it so users with movie most in common with the input will have priority
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
### TO SUMMARIZE THIS LINE ABOVE
# sorted() does its thing, order it in reverse order due to reverse=True
# The lambda sketchiness means that for each row it will select the 1st index, which is user id
# The len serves to count how many users with that id is in common, and that's what it sorts it by
# This, it groups users who rated movies the most in common with the input getting priority

print(userSubsetGroup[0:3]) # First three tuples of user ids with the most in common with userInput
# As you can notice, before the tuple, it outputs the user_id

# Pearson Correlation:
# Returns r between 1 and -1, with 1 meaning exactly identical and -1 be exactly opposite
# Vectors are INVARIANT to scaling, pearson(X, Y) == pearson(X, 2 * Y + 3)

# Select a subset of the userSubsetGroup to go through, since we ordered it from most relevant to least
# We just select the first 100 distinct user id's
userSubsetGroup = userSubsetGroup[0:100]

# FIND PEARSON CORRELATIONS
# Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
pearsonCorrelationDict = {}

# For every user group in our subset
for name, group in userSubsetGroup:
    # Let's start by sorting the input and current user group so the values aren't mixed up later on
    # The name is the userID and the group is everything that is the tuple
    # (75, userId  movieId  rating) name, group as you can see
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    # Get the N for the formula
    nRatings = len(group)
    # Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())] # This is the user data
    # And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist() # User rating
    # Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist() # Database user rating
    # Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i ** 2 for i in tempRatingList]) - pow(sum(tempRatingList), 2) / float(nRatings)
    Syy = sum([i ** 2 for i in tempGroupList]) - pow(sum(tempGroupList), 2) / float(nRatings)
    Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList) * sum(tempGroupList) / float(
        nRatings)

    # If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy / sqrt(Sxx * Syy)
    else:
        pearsonCorrelationDict[name] = 0

print("\nPearson correlations for user_id's:", pearsonCorrelationDict.items())

pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
print("\nPearson correlations in a table:", pearsonDF.head())

# Output the top 50 similar or neighbor users
topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
print("\nTop 50 similar users:", topUsers.head())

# Merge top users with ratings
topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
print(topUsersRating.head())

#Multiplies the similarity by the user's ratings to find weights so we can add them together
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
print(topUsersRating.head())

#Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
print(tempTopUsersRating.head())

# Create the Movie Recs
#Creates an empty dataframe
recommendation_df = pd.DataFrame()
#Now we take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df.head()

# TOP 10 MOVIES
recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
print(recommendation_df.head(10))

# WITH TITLES
print(movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())])