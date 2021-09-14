import joblib
import pandas as pd
import re
import numpy as np

def clean_html(html):
    """
    Applies some pre-processing on the given text.

    Steps :
    - Removing HTML tags
    - Removing punctuation
    - Lowering text
    """
    OUTPUT = []
    for text in html:
        # remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # remove the characters [\], ['] and ["]
        text = re.sub(r"\\", "", text)
        text = re.sub(r"\'", "", text)
        text = re.sub(r"\"", "", text)
        text = re.sub(r"\xa0", "", text)
        text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

        OUTPUT.append(text)

    return OUTPUT


def load_model(model_filename='../Models/model.pkl'):
    """
    Loads and returns the pretrained model
    """
    model = joblib.load(model_filename)
    print("Model loaded")
    return model


def predict(input_data,model,seuil_max=0.7):
    """
    Clean the input Data and returns the predicted values
    """

    #clean the html code
    input_data=clean_html(input_data)

    #find the id of the category predicted (with the max proba)
    id = model.predict(input_data)
    id= [True if i=='t' else False for i in id]
    # find the max probability
    proba = np.max(model.predict_proba(input_data), axis=1)

    # variable to see if checking is needed
    need_check = [True if (i < seuil_max) else False for i in proba]
    response = {
        "us_only": id,
        "proba": proba,
        "need check": need_check,
    }

    return response

if __name__ == "__main__":

    model = load_model()
    df=pd.read_csv('../Data/job_export.csv')
    job_descriptions=df['general_description']
    prediction=predict(job_descriptions,model=model)
    print(prediction)