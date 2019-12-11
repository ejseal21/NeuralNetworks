from nltk.stem import PorterStemmer

def stem(corpus, strings):
    '''
    Takes in word vectors and word strings, returns a new set of
    word vectors and word strings after stemming all of the words
    '''
    new_corpus = corpus
    ps = PorterStemmer()
    new_strings = []
    for i in range(len(strings)):
        #get the stems
        new_strings.append(ps.stem(strings[i]))
        
        #if stemming the word changed it
        if new_strings[i] != strings[i]:
            #find every instance of the word in the corpus
            for j in range(len(corpus)):
                for k in range(len(corpus[j])):
                    if corpus[j][k] == strings[i]:
                        #and replace it with the new one in the new_corpus
                        new_corpus[j][k] = new_strings[i]
    
    #decrease the size of strings so that we can pass unique_words in and out
    i = 0
    while len(new_strings) > i:
        if strings[i] != new_strings[i]:
            if new_strings[i] in new_strings[:i] or new_strings[i] in new_strings[i+1:]:
                new_strings.pop(i)
                i -= 1
        i += 1
    return new_corpus, new_strings