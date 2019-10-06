# Twitter-NLP-analysis
Sentiment analysis of tweets with NLP

**Dataset:**   
The dataset is collected in Spring 2014 from Twitter. It is not included in public repo due to Twitter terms of service.  

**Data preprocessing:**  
   classifySemevalTweets.py  
-  classifies data in 3 buckets - pos, neg and neu 
-  Convert to lower case 
-	 Remove URLs since they don’t contribute to sentiment of tweet 
-	 Remove usernames  
-	 Remove ‘#’ from hashtags 

**Generation of features:**
To generate features, first I used ‘bag of words’ model which considers all the words in the text regardless of grammar or order. We are using most common 2000 words to build features. They are appended with ‘V_’ to indicate vocabulary feature. 

**Processing:**
-  Naive Bayes classifier 
-  Stopword removal 
-  Subjectivity lexicon 
-  Opinion lexicon 
-  SGDClassifier 
-  Nearest neighbors 
-  Linear SVM 
-  Decision trees 
-  Random Forest 

