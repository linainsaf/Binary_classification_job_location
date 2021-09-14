#import libraries
import pandas as pd
import re

def select_column(df,input_data="name",labels="tag_id",date_time="publication_date"):
    """
    Select the main columns for our learning.

       """
    df[['input_data','labels','date_time']]=df[[input_data,labels,date_time]]
    return df[['input_data','labels','date_time']]

#croping datetime:
def crop_data(df,starting_date=None,ending_date=None):
    """
    Choose a time interval.

     """
    df['date_time'] = df.date_time.apply(lambda a: pd.to_datetime(a).date())
    if (ending_date==None):
        start_date = pd.to_datetime(starting_date, utc=True)
        df = df.loc[(df['date_time'] > start_date)]
    elif (starting_date==None):
        end_date = pd.to_datetime(ending_date, utc=True)
        df = df.loc[(df['date_time'] < end_date)]
    else:
        start_date = pd.to_datetime(starting_date, utc=True)
        end_date = pd.to_datetime(ending_date, utc=True)
        df = df.loc[(df['date_time'] > start_date)]
        df = df.loc[(df['date_time'] < end_date)]

    return df

def select_labels(df,labels):
    """
   selecting the labels that we want.

    Steps :
    - Drop Nans
    - Removing duplicates
    - Remove labels that we don't need
    """
    # remove none tagged job offers
    df.dropna(subset=['labels'], inplace=True)
    #remove duplicates
    df.drop_duplicates(subset=['input_data'])
    #select labels
    df=df.loc[df['labels'].isin(list(labels))]
    return df

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


if __name__ == "__main__":

    LABELS = ['t','f']

    #import dataframe and select the columns needed
    df=select_column(df= pd.read_csv('../Data/job_export.csv'),
                     input_data="general_description",labels="us_only",
                     date_time="publication_date")

    #drop the categories that we dont need and the NAN
    df=select_labels(df,labels=LABELS)

    df['input_data']=clean_html(df.input_data)


    df.to_csv('../Data/data_cleaned.csv', index=False)
