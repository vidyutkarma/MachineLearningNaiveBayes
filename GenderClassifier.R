#Loading the package.
if(!require("tm")) 
{
  install.packages("tm")
  library(tm)
}

if(!require("e1071")) 
{
  install.packages("e1071")
  library(e1071)
}

if(!require("gmodels")) 
{
  install.packages("gmodels")
  library(gmodels)
}
#First let us load the data. Since, the statuses and profiles are in a separate files, We need create our dataset (comprising of text and gender) 
# by reading each entry in the profile file and then reading the corresponding text file (which has the same name as the userid in 
# in profile file)

#loading the gender information from the profile file
genderProfile <- read.csv("~/Desktop/Study/Books/MachineLearning/Independent/Gender/TCSS555/Train/Profile/Profile.csv", header = TRUE, colClasses = c("NULL", NA, "NULL", NA, "NULL", "NULL", "NULL", "NULL", "NULL"))

#Create statuses file names. We shall first mention the path to the files and then use paste0 to create full file names

textPath <- "~/Desktop/Study/Books/MachineLearning/Independent/Gender/TCSS555/Train/Text/"
textFileNames <- paste0(textPath, genderProfile$userid, ".txt")

#Loading the text data in gender_profile data set. The function returnText has been defined in the function section
genderProfile$text <- sapply(textFileNames, returnText, USE.NAMES = F)

#Since profile id is no longer required, we shall drop that column and re-orient our columns to have text as the first column
genderProfile <- data.frame(text = genderProfile$text, gender = factor(genderProfile$gender), stringsAsFactors = FALSE)

#Create a corpus of text using the Corpus command
textCorpus <- Corpus(VectorSource(genderProfile$text))

#Cleaning of text using tm_map()
textCorpusClean <- tm_map(textCorpus,
                              content_transformer(function(x) iconv(x, to='UTF-8-MAC', sub='byte')),
                              mc.cores=1
)
textCorpusClean <- tm_map(textCorpusClean, content_transformer(tolower), mc.cores=1) #Convert to lower case

textCorpusClean <- tm_map(textCorpusClean, removeNumbers, mc.cores=1) #Remove numbers

textCorpusClean <- tm_map(textCorpusClean, removeWords, stopwords(), mc.cores=1) #Remove fillers like to, and, but etc..

textCorpusClean <- tm_map(textCorpusClean, removePunctuation, mc.cores=1) # Remove punctuation

textCorpusClean <- tm_map(textCorpusClean, stripWhitespace, mc.cores=1) # The previous steps left spaces while cleaning out the text. This will leave a single space between words

#Now split the text into individual componenets using tokenization.
textTDM <- DocumentTermMatrix(textCorpusClean) # This returns a sparse matrix in which the rows indicate the documents and the columns indicate the terms (words). Each cell stores a number indicating the number of times the word appeared in the row.

#Prepare train and test data. We shall split the data in 70:30 ratio. Before doing that, we must ensure that training and test set have uniform data

prop.table(table(genderProfile$gender[1:6650]))

# 0        1 
# 0.422406 0.577594 
prop.table(table(genderProfile$gender[6651:9500]))

# 0         1 
# 0.4238596 0.5761404 

#Since the data is uniformly distributed, we can straightaway split the data in 70:30 ratio

#Raw data
textRawTrain <- genderProfile[1:6650, ]
testRawTest <- genderProfile[6651:9500, ]

#Document term matrix:
textTDMTrain <- textTDM[1:6650, ]
textTDMTest <- textTDM[6651:9500, ]

#Corpus data:
textCorpusTrain <- textCorpusClean[1:6650]
textCorpusTest <- textCorpusClean[6651:9500]

#Since all the words in the sparse matrix are not useful for classification, we shall use words which occur atleast 10 times

textDict <- findFreqTerms(textTDMTrain, 20) # with 10 and 20, the accuracy was 70% for both non-lapace and laplace estimator

#It seems words that appear in test set and not in training set are causing problems. We shall try to take them out

#removeWordsInt <- c("audible")
#textDict <- textDict[!is.element(textDict, removeWordsInt)]

textTrain <- DocumentTermMatrix(textCorpusTrain, list(dictionary = textDict))
textTest <- DocumentTermMatrix(textCorpusTest, list(dictionary = textDict))

#Typically Naive Bayes is trained on categorical features. We shall convert training and test data to factors

textTrain <- apply(textTrain, MARGIN = 2, convertCount)
textTest <- apply(textTest, MARGIN = 2, convertCount)

#We shall use e1071 package 
library(e1071)

genderClassifier <- naiveBayes(textTrain, textRawTrain$gender) # Training the classifier

genderPred <- predict(genderClassifier, textTest) # making the prediction on test data

#Comparision of predicted against the actual

library(gmodels)

CrossTable(genderPred, testRawTest$gender, prop.chisq = FALSE, prop.t = FALSE, dnn = c("predicted", "actual"))


#Cell Contents
# |-------------------------|
#   |                       N |
#   |           N / Row Total |
#   |           N / Col Total |
#   |-------------------------|
#   
#   
#   Total Observations in Table:  2850 
# 
# 
# | actual 
# predicted |         0 |         1 | Row Total | 
#   -------------|-----------|-----------|-----------|
#   0 |       804 |       445 |      1249 | 
#   |     0.644 |     0.356 |     0.438 | 
#   |     0.666 |     0.271 |           | 
#   -------------|-----------|-----------|-----------|
#   1 |       404 |      1197 |      1601 | 
#   |     0.252 |     0.748 |     0.562 | 
#   |     0.334 |     0.729 |           | 
#   -------------|-----------|-----------|-----------|
#   Column Total |      1208 |      1642 |      2850 | 
#   |     0.424 |     0.576 |           | 
#   -------------|-----------|-----------|-----------|

#As can be seen above, the accuracy is around 70.21%. Lets use laplace estimator

genderClassifierLaplace <- naiveBayes(textTrain, textRawTrain$gender, laplace = 1) # Training the classifier with laplace = 1

genderPredLaplace <- predict(genderClassifierLaplace, textTest) # making the prediction on test data 

CrossTable(genderPredLaplace, testRawTest$gender, prop.chisq = FALSE, prop.t = FALSE, dnn = c("predicted", "actual"))


# Cell Contents
# |-------------------------|
#   |                       N |
#   |           N / Row Total |
#   |           N / Col Total |
#   |-------------------------|
#   
#   
#   Total Observations in Table:  2850 
# 
# 
# | actual 
# predicted |         0 |         1 | Row Total | 
#   -------------|-----------|-----------|-----------|
#   0 |       796 |       441 |      1237 | 
#   |     0.643 |     0.357 |     0.434 | 
#   |     0.659 |     0.269 |           | 
#   -------------|-----------|-----------|-----------|
#   1 |       412 |      1201 |      1613 | 
#   |     0.255 |     0.745 |     0.566 | 
#   |     0.341 |     0.731 |           | 
#   -------------|-----------|-----------|-----------|
#   Column Total |      1208 |      1642 |      2850 | 
#   |     0.424 |     0.576 |           | 
#   -------------|-----------|-----------|-----------|

#As can be seen above, the accuracy has marginally decreased to 70.07%  
############################# Functions #######################
#Create a function to return all the lines in the text file as a single line.

returnText <- function(x) {
  paste(readLines(x, warn = F), collapse = ' ')
}

convertCount <- function(x) {
  x <- ifelse(as.numeric(x) > 0, "yes", "no")
  x <- factor(x, levels = c("no", "yes"))
}