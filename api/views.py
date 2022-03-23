from pickle import GET
from rest_framework.response import Response
from rest_framework.decorators import api_view

# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# import joblib for model naive bayes
import joblib 

# steaming function with sastrawi
def text_sastrawi(text, remove_stop_words=True, lemmatize_words=True):
    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    # stemming process
    text = stemmer.stem(text)

    return text

# load the nb model
with open("models/judul_model_nb.pkl", "rb") as f:
    model = joblib.load(f)

# load the tfid coverter
with open("models/judul_model_tfid.pkl", "rb") as f:
    vectorizer = joblib.load(f)
    
@api_view(['GET'])
def getData(request):
    person = {'tes' : 'zufar'}
    return Response(person)

@api_view(['GET'])
def NBTitle(req):
    # Rest of the method
    judul = req.query_params.get('judul')
    
    # clean the review
    judul_stemming = text_sastrawi(judul)

    # # tfid
    judul_tfid = vectorizer.transform([judul_stemming])

    # perform prediction
    prediction = model.predict(judul_tfid)
    output = int(prediction)
    probas = model.predict_proba(judul_tfid)

    output_probability = "{:.2f}".format(float(probas[:, output]))
    
    # output dictionary
    sentiments = {
        0: "Umum", 
        1: "Filsafat dan Psikologi",
        2: "Agama",
        3: "Sosial",
        4: "Bahasa",
        5: "Sains dan Matematika",
        6: "Teknologi",
        7: "Seni dan Rekreasi",
        8: "Literartur dan Sastra",
        9: "Sejarah dan Geografi"
    }
    
    # show results
    result = {
        "judul_origin": judul,
        "judul_stemming": judul_stemming,
        "prediction": sentiments[output],
        "probability": output_probability,
        "probability_class": {
            "{:.2f}".format(float(probas[:, 0])) : "Umum", 
            "{:.2f}".format(float(probas[:, 1])) : "Filsafat dan Psikologi",
            "{:.2f}".format(float(probas[:, 2])) : "Agama",
            "{:.2f}".format(float(probas[:, 3])) : "Sosial",
            "{:.2f}".format(float(probas[:, 4])) : "Bahasa",
            "{:.2f}".format(float(probas[:, 5])) : "Sains dan Matematika",
            "{:.2f}".format(float(probas[:, 6])) : "Teknologi",
            "{:.2f}".format(float(probas[:, 7])) : "Seni dan Rekreasi",
            "{:.2f}".format(float(probas[:, 8])) : "Literartur dan Sastra",
            "{:.2f}".format(float(probas[:, 9])) : "Sejarah dan Geografi"
        }
    }
    return Response(result)
