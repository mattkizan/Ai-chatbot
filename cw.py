
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk

from nltk.corpus import stopwords

from nltk import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

import csv

from sklearn.model_selection import train_test_split

import warnings

from dateutil import parser
import pandas as pd

warnings.filterwarnings("ignore")


nltk.download('maxent_ne_chunker') ####################### may be required ###########################################
nltk.download('words')


def stemmed_words(doc): # stemms the word when using count vectoriser or tf-idf vectoriser
    p_stemmer = PorterStemmer()
    analyzer = CountVectorizer().build_analyzer()
    return (p_stemmer.stem(w) for w in analyzer(doc))


def Similarity_based_intent(userInput, filename): # retrieves answer from the question and answer pair that is the most similar to the user input
    vect = TfidfVectorizer(lowercase=True, stop_words=stopwords.words('english'), analyzer=stemmed_words, use_idf=True,
                           sublinear_tf=True)

    Questions = []
    Answers = []

    with open(filename, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for line in csv_reader:
            Questions.append(line['question'])
            Answers.append(line['answer'])

    database = vect.fit_transform(Questions)
    userResponse = vect.transform([userInput])

    similarities = cosine_similarity(userResponse, database)

    # Find the index of the most similar sentence
    most_similar_index = similarities.argmax()

    # Get the confidence score of the most similar sentence
    similarity_confidence_score = similarities[0, most_similar_index]

    return Answers[most_similar_index], similarity_confidence_score


def entity_recognition(userinput): # function extracts Persons or names from the string: user input 
    string = "name is " + userinput

    word = word_tokenize(string)
    tagged = nltk.pos_tag(word)

    entities = nltk.ne_chunk(tagged)

    names = []

    for entity in entities:

        if isinstance(entity, nltk.Tree) and entity.label() == 'PERSON':
            name = ' '.join([word[0] for word in entity.leaves()])
            names.append(name)

    if not names:
        names.append(userinput)

    return names[0]


def intent_recognition(userinput):  # classifer takes user input and predicts class label for intent
    # labels = []
    # data = []
    #
    # with open('intent.csv', 'r') as csv_file:
    #     csv_reader = csv.DictReader(csv_file)
    #
    #     for line in csv_reader:
    #         labels.append(line['intent'])
    #         data.append(line['question'])

    df = pd.read_csv('intent.csv')

    data = df['question']
    labels = df['intent']

    X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels, test_size=0.2, random_state=1)

    count_vect = CountVectorizer(lowercase=True, stop_words=stopwords.words('english'), analyzer=stemmed_words)
    X_train_counts = count_vect.fit_transform(X_train)

    tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(X_train_counts)
    X_train_tf = tfidf_transformer.transform(X_train_counts)

    clf = LogisticRegression(random_state=0).fit(X_train_tf, y_train)

    processed_newdata = count_vect.transform(userinput)

 
    processed_input = tfidf_transformer.transform(processed_newdata)

    classes = [0, 1, 2, 3, 4]

    pred_prob = clf.predict_proba(processed_input)
    intent = pred_prob.argmax()

    return classes[intent], pred_prob[0, intent]


def extract_location(text): # extracts locations from string : text
    words = word_tokenize(text)
    tagged_words = nltk.pos_tag(words)
    entities = nltk.ne_chunk(tagged_words)

    location = []

    for entity in entities:

        if isinstance(entity, nltk.Tree) and entity.label() == 'GPE':
            location_name = ' '.join([token[0] for token in entity.leaves()])

            location.append(location_name)

    return location


def check_return_or_single_trip(user_input):  # function check to see if  the word return or single is in the  string:
    # user inpuit
    trip_options = ["return", "single"]

    user_input_lower = user_input.lower()

    words = word_tokenize(user_input_lower)

    single_or_return_response = []
    for word in words:
        if word in trip_options:
            single_or_return_response.append(word)

    return single_or_return_response


def extract_date(user_input):  # function extracts date from string : user input
    words = word_tokenize(user_input)

    # extracting date using inbuilt func.
    dates = []
    for word in words:
        try:
            res = parser.parse(word, fuzzy=True)
            date = res.strftime("%d-%m-%Y")
            dates.append(date)
        except ValueError:
            pass

    return dates


def tranaction_intent_recognition(userinput):  # classifier takes user input and predicts class label for intent
    # labels = []
    # data = []
    #
    # with open('transactions.csv', 'r') as csv_file:
    #     csv_reader = csv.DictReader(csv_file)
    #
    #     for line in csv_reader:
    #         labels.append(line['intent'])
    #         data.append(line['input'])
    df = pd.read_csv('/Users/matt/Desktop/transactions.csv')

    data = df['input']
    labels = df['intent']
    X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels, test_size=0.2, random_state=1)

    count_vect = CountVectorizer(lowercase=True, stop_words=stopwords.words('english'), analyzer=stemmed_words)
    X_train_counts = count_vect.fit_transform(X_train)

    tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(X_train_counts)
    X_train_tf = tfidf_transformer.transform(X_train_counts)

    clf = LogisticRegression(random_state=0).fit(X_train_tf, y_train)

    processed_newdata = count_vect.transform(userinput)

    # document-term for training

    # Weighting
    processed_input = tfidf_transformer.transform(processed_newdata)

    # Make predictions on the test set

    classes = [1, 2, 3, 4]

    pred_prob = clf.predict_proba(processed_input)
    trans_intent = pred_prob.argmax()

    return classes[trans_intent], pred_prob[0, trans_intent]


def transactions(name):  # transaction function used to start transactional dialogue
    trip_type_response = ""
    state = 0
    arrival = ""
    departure = ""
    trip_type = " "
    date = []
    # user prompted to enter destination
    print("chatbot: Welcome to the Flight Booking System! Firstly, Please enter the departure location and the arrival "
          "destination ")
    while True:  # while loop keeps taking user input, until user enters STOP

        userinput = input(f"{name}:  ")

        trans_intent, trans_confidence = tranaction_intent_recognition(
            [userinput])  # intent is classifed from user input as well as confidence score

        if trans_intent == 1 and trans_confidence > 0.5 and state == 0:  # if intent is: providing locations and if
            # location is not already provided

            locations = extract_location(userinput)  # extracts locations from user input

            if len(locations) == 2:  # checks to see if two locations were given
                state = 1  # context tracking - prevents user to provide locations again, moves onto the trip type process of the transaction
                arrival = locations[0]
                departure = locations[1]
                print(f"Chatbot: Okay. Setting the arrival and departure locations to be {arrival} and {departure}")
                print(f"Chatbot: Alright {name}. Would that be a return trip or a single trip?")

            if len(locations) != 2:
                print("Chatbot: Please provide two departure location and arrival destination in upper case format.")

        if trans_intent == 2 and trans_confidence > 0.5 and state == 1:  # if intent is: specifying whether the trip is single or return and if trip response is not already given

            trip_type_response = check_return_or_single_trip(
                userinput)  # check to see if user input contains 'single' or 'return'

            if len(trip_type_response) == 1:
                state = 2

                print(f"Chatbot: Got it, {trip_type_response[0]} trip. Can you specify the dates please.")

            if len(trip_type_response) != 1:
                print("Chatbot: Please provide return or single as your input. ")

        if trans_intent == 3 and trans_confidence > 0.5 and state == 2:  # if intent is : providing dates and if
            # dates is already given

            date = extract_date(userinput)  # extract dates from user input

            if len(date) == 1:
                print(f"Departure date set to {date[0]}")
                print(
                    f"Chatbot: {name} Your {trip_type_response[0]} flight from {departure} to {arrival} on the {date[0]} has "f"been booked!")  # confirmation mesasage
                print("Chatbot: returning home..")
                break  # exits loop - transaction finsihed

            if len(date) == 2:
                print(f"Arrival and Departure date set to {date[0]} and {date[1]}")
                print(
                    f"Chatbot: {name} Your {trip_type_response[0]} trip flight from {departure} to {arrival} , {date[0]} to {date[1]} has been booked!")  # confirmation message
                print("Chatbot: returning home..")
                break  # exits loop - transaction finsihed

            else:
                print("Chatbot: Please specify the dates in the format of dd-mm-year")

        if userinput.strip() == "STOP":  # if user enters stop , transaction will be cancelled
            print("Chatbot: exiting transaction")
            break
        if trans_intent == 4 and trans_confidence > 0.5:
            break

        if trans_confidence < 0.5 and state == 0:  # if intent is not applicable to the transactional dialogue
            print("Chatbot: Please provide two departure location and arrival destination in upper case format,")
        if trans_confidence < 0.5 and state == 1:  # if intent is not applicable to the transactional dialogue
            print("Chatbot: Please provide return or single as your input. ")
        if trans_confidence < 0.5 and state == 2:  # if intent is not applicable to the transactional dialogue
            print("Chatbot: Please specify the dates in the format of dd-mm-year.")


# main function - starts the chatbot - where the core features get called
if __name__ == '__main__':

    stop = False
    name_not_stored = True
    name = " "
    # while loop keeps taking user input, until user enters STOP
    while True:
        # if name is not stored, then the chatbot will ask the user to enter their name
        if name_not_stored:
            print("Chatbot: Hello, Please enter your name in uppercase")
            name_response = input("You:  ")
            # entity_recognition function extracts named entity person from the user input
            name = entity_recognition(name_response)
            print(f"Chatbot: Hello {name}")
            name_not_stored = False

            print("Bot functionality: small talk, request user name, make a travel "
                  "booking "
                  "and answer questions regarding travel and tourism.")
            print("Chatbot: Enter your query , or enter STOP to quit , and press return : ")

        # prompt to request the user to enter a query
        query = input(f"{name}:  ")

        # send user query to classify intent and return intent class and confidence level
        intent, confidence = intent_recognition([query])

        # if statement for each intent class with a confidence threshold
        # intent is 0 if user asks the system to recall their name
        if intent == 0 and confidence > 0.5:
            print(f"Chatbot: Your name is {name}")

        # intent is 1 if user intent is to  make a booking
        if intent == 1 and confidence > 0.5:
            transactions(name)  # transaction function is called

        # intent is 2 if the user wants to make small talk
        if intent == 2 and confidence > 0.5:
            response, confidence_score = Similarity_based_intent(query, 'smalltalk.csv')  # function matches user
            # input to most similar question and answer pair in smalltalk.csv and returns the response for the most
            # similar question

            if confidence_score > 0.5:  # threshold for the similarity score
                print(f"Chatbot: {response}")
            else:
                print("Chatbot: I am sorry I did no not have an answer to that, maybe try another question?")

        # intent 3 is question and answer retrieval
        if intent == 3 and confidence > 0.5:

            response, confidence_score = Similarity_based_intent(query, 'qa.csv')  # function finds most similar
            # question and answer pair to user's query and returns the answer

            if confidence_score > 0.5:
                print(f"Chatbot: {response}")
            else:
                print("Chatbot: I am sorry I did no not have an answer to that, maybe try another question?")

        # intent is 4 is user request help to find out the functionality of the chatbot
        if intent == 4 and confidence > 0.5:
            print("Chatbot: You can make a travel booking, ask questions relating to travel and tourism, request your "
                  "name and participate in small talk")
            print("OR enter STOP to quit")

        # if user input is STOP the program will stop.
        if query.strip() == "STOP":
            print("Chatbot: Bye!")
            break
        # if confidence from intent recognition is below the threshold, reprompt message me will be displayed
        if confidence < 0.5:
            print("Chatbot: I am sorry I did not get not get that, Can you be more specific?")
