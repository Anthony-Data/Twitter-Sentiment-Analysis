# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:28:24 2021

@author: Antho
"""

from textblob import TextBlob
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer


consumer_key = 'btBoEyu4VKREsOqPMwze0QWSY'
consumer_secret = '4r9SmYil6NjjSBENwK4qBW0ci5s6RisTTw3zZx6ErShoh625X0'
access_key= '1373662865496887300-mTlKcyDR2XQSfxhxw7TcYJvP0ypGKj'
access_secret = 'rzId8IIe4Y0ThRCn9ha6GzCPNXAVohVevlSUXwhXHiGeQ'


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


def percentage(part,whole):
    return 100 * float(part)/float(whole) 


keyword = input("Please enter keyword or hashtag to search: ")
noOfTweet = int(input("Please enter how many tweets to analyze: "))

tweets = tweepy.Cursor(api.search, q=keyword, lang ='en').items(noOfTweet)

positive  = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []

for tweet in tweets:
        tweet_list.append(tweet.text)
        analysis = TextBlob(tweet.text)
        score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        comp = score['compound']
        polarity += analysis.sentiment.polarity
        if neg > pos:
            negative_list.append(tweet.text)
            negative += 1
        elif pos > neg:
            positive_list.append(tweet.text)
            positive += 1
        elif pos == neg:   
            neutral_list.append(tweet.text)
            neutral += 1
            
positive = percentage(positive, noOfTweet)
negative = percentage(negative, noOfTweet)
neutral = percentage(neutral, noOfTweet)
polarity = percentage(polarity, noOfTweet)
positive = format(positive, '.1f')
negative = format(negative, '.1f')
neutral = format(neutral, '.1f')
    
tweet_list = pd.DataFrame(tweet_list)
neutral_list = pd.DataFrame(neutral_list)
negative_list = pd.DataFrame(negative_list)
positive_list = pd.DataFrame(positive_list)

print("total number: ",len(tweet_list))
print("positive number: ",len(positive_list))
print("negative number: ", len(negative_list))
print("neutral number: ",len(neutral_list)) 

labels = ['Positive ['+str(positive)+'%]' , 'Neutral ['+str(neutral)+'%]','Negative ['+str(negative)+'%]']
sizes = [positive, neutral, negative]
colors = ['yellowgreen', 'blue','red']
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title("Sentiment Analysis Result for keyword=  "+keyword+"" )
plt.axis('equal')
plt.show()
   
tweet_list.drop_duplicates(inplace = True)
tw_list = pd.DataFrame(tweet_list)
tw_list["text"] = tw_list[0]

#Cleaning Text (RT, Punctuation etc)
remove_rt = lambda x: re.sub('RT @\w+: '," ",x)
rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)
tw_list["text"] = tw_list.text.map(remove_rt).map(rt)
tw_list["text"] = tw_list.text.str.lower()

#Calculating Negative, Positive, Neutral and Compound values
tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
for index, row in tw_list['text'].iteritems():
            score = SentimentIntensityAnalyzer().polarity_scores(row)
            neg = score['neg']
            neu = score['neu']
            pos = score['pos']
            comp = score['compound']
            if neg > pos:
                tw_list.loc[index, 'sentiment'] = "negative"
                
            elif pos > neg:
                tw_list.loc[index, 'sentiment'] = "positive"
                
            else:
                        tw_list.loc[index, 'sentiment'] = "neutral"
                        
                        

#print(id_list)
    
    #Creating new data frames for all sentiments (positive, negative and neutral)  
tw_list_negative = tw_list[tw_list["sentiment"]=="negative"]
tw_list_positive = tw_list[tw_list["sentiment"]=="positive"]
tw_list_neutral = tw_list[tw_list["sentiment"]=="neutral"]


tw_list.drop('polarity', inplace=True, axis=1)
tw_list.drop('subjectivity', inplace=True, axis=1)
tw_list.drop(0, inplace=True, axis=1)

def work():
    tw_list.to_csv('TweetsList.csv')
    
    
if __name__ == '__main__':

    work()

