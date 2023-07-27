import joblib
from transformers import pipeline
from snowflake.snowpark import Session
from snowflake.snowpark.functions import pandas_udf
from snowflake.snowpark.types import StringType, PandasSeriesType
import cachetools
import sys
import pandas as pd


from transformers import pipeline

sentiment_analysis_model = pipeline("sentiment-analysis", model='distilbert-base-uncased-finetuned-sst-2-english')
results=sentiment_analysis_model("I love the way we implement ML models at Infostrux")



joblib.dump(sentiment_analysis_model, 'sentiment-analysis.joblib')


session = Session.builder.configs({
  "account": "your_account",
  "user": "your_user",
  "password": "password",
  "role": "SYSADMIN",
  "warehouse": "SNOWPARK_DEMO",
  "database": "ML_DEMO_DB",
  "schema": "ML_DEMO_SCHEMA"
}
 ).create()
session.file.put(
   'sentiment-analysis.joblib',
   stage_location = f'@ML_DEMO_DB.ML_DEMO_SCHEMA.my_pretrained_models_stage',
   overwrite=True,
   auto_compress=False
)


@cachetools.cached(cache={})
def read_model():
   import_dir = sys._xoptions.get("snowflake_import_directory")
   if import_dir:
       # Load the model
       return joblib.load(f'{import_dir}/sentiment-analysis.joblib')


@pandas_udf(  
       name='ML_DEMO_DB.ML_DEMO_SCHEMA.get_sentiment',
       session=session,
       is_permanent=True,
       replace=True,
       imports=[
           f'@ML_DEMO_DB.ML_DEMO_SCHEMA.my_pretrained_models_stage/sentiment-analysis.joblib'
       ],
       input_types=[PandasSeriesType(StringType())],
       return_type=PandasSeriesType(StringType()),
       stage_location='@ML_DEMO_DB.ML_DEMO_SCHEMA.my_pretrained_models_stage',
       packages=['cachetools==4.2.2', 'transformers==4.14.1']
   )

def get_setiment(sentences):   
   # Load the sentiment analysis model from stage
   # using the caching mechanism
   sentiment_analysis_model = read_model()

    # Apply the model
   predictions = []
   for sentence in sentences:
       result = sentiment_analysis_model(sentence)
       predictions.append(result)
   return pd.Series(predictions)

