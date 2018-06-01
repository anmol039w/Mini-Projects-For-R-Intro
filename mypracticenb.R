library(tm)

#input of file

df <- read.csv("C://Users//Anmol//Desktop//Classification//sms_spam.csv",stringsAsFactors = FALSE)

#anlaysing the data
head(df)
str(df)
summary(df)

df$type <- factor(df$type)
#getting count of ham and spam

table(df$type)

#creating corpus

sms_corpus <- VCorpus(VectorSource(df$text))

#inspecting the corporus #list indexing
sms_corpus[[1]]
sms_corpus[[2]]

#viewing the metadata
meta(sms_corpus[[1]])
meta(sms_corpus[[2]])
meta(sms_corpus[[3]])

#viewing the characters
as.character(sms_corpus[[1]])

lapply(sms_corpus[[1:2]], as.character)

#cleaning the data #upper to lower 

sms_corpus_clean <- tm_map(sms_corpus,content_transformer(tolower))

#removing numbers from the messages

sms_corpus_clean <- tm_map(sms_corpus_clean,removeNumbers)


#removing the stop words
sms_corpus_clean <- tm_map(sms_corpus_clean,removeWords,stopwords())

#removing punctuations
sms_corpus_clean <- tm_map(sms_corpus_clean,removePunctuation)

#steeming for removing the text with similar base class

library(SnowballC)

#testing the function
wordStem(c("learning","learn","learned","learner"))

sms_corpus_clean <- tm_map(sms_corpus_clean,stemDocument)

#stripping the whitespaces

sms_corpus_clean <- tm_map(sms_corpus_clean,stripWhitespace)

#creating a word matrix
sms_dttm <- DocumentTermMatrix(sms_corpus_clean)

#creating the sample
str(sms_dttm)

sms_train <-sms_dttm[1:4169]
sms_test <- sms_dttm[4170:5574]

#adding labels
sms_train_label <- df[1:4169,]$type
sms_test_lable <- df[4170:5574,]$type

#seeing the ditribution
table(sms_test_label)
table(sms_train_lable)

#visualising the trained dataset
library(wordcloud)

wordcloud(sms_corpus_clean,min.freq = 50,random.order = FALSE)

#comparing cloud of spam and ham

spam <- subset(df,type=="spam")
ham <- subset(df,type=="ham")

wordcloud(spam$text,max.words = 100,scale = c(3,0.5))
wordcloud(ham$text,max.words = 100,scale = c(3,0.5))

#creating frequent words
findFreqTerms(sms_train,5)
