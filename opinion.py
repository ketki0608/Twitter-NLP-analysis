import nltk
import re

def readOpinion(path):
    poslexicon = open(path+'/positive-words.txt', 'r')
    neglexicon = open(path+'/negative-words.txt', 'r', encoding="latin-1")
    sldict = { }
    for line in poslexicon:
        word = re.sub('\\n', '', line)
        polarity = 'positive'
        # put a dictionary entry with the word as the keyword
        #     and a list of the other values
        sldict[word] = [polarity]
    for line in neglexicon:
        word = re.sub('\\n', '', line)
        polarity = 'negative'
        # put a dictionary entry with the word as the keyword
        #     and a list of the other values
        sldict[word] = [polarity]
    return sldict
