import joblib
from snowflake.snowpark import Session
from snowflake.snowpark.functions import pandas_udf
from snowflake.snowpark.types import StringType, PandasSeriesType
import cachetools
import sys
import pandas as pd
from transformers import pipeline

summarize = pipeline('summarization', model='philschmid/bart-large-cnn-samsum')

 

paragraph ="""The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.

The field of AI research was founded at a workshop held on the campus of Dartmouth College, USA during the summer of 1956.[1] Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true.[2]

Eventually, it became obvious that commercial developers and researchers had grossly underestimated the difficulty of the project.[3] In 1974, in response to the criticism from James Lighthill and ongoing pressure from congress, the U.S. and British Governments stopped funding undirected research into artificial intelligence, and the difficult years that followed would later be known as an "AI winter". Seven years later, a visionary initiative by the Japanese Government inspired governments and industry to provide AI with billions of dollars, but by the late 1980s the investors became disillusioned and withdrew funding again.

Investment and interest in AI boomed in the first decades of the 21st century when machine learning was successfully applied to many problems in academia and industry due to new methods, the application of powerful computer hardware, and the collection of immense data sets."""



output=summarize(paragraph)

print(output)

joblib.dump(summarize, 'summarization.joblib')


session = Session.builder.configs({
  "account": "yur_snowflake_account",
  "user": "your_user",
  "password": "password",
  "role": "SYSADMIN",
  "warehouse": "SNOWPARK_DEMO",
  "database": "ML_DEMO_DB",
  "schema": "ML_DEMO_SCHEMA"
}
 ).create()
session.file.put(
   'summarization.joblib',
   stage_location = f'@ML_DEMO_DB.ML_DEMO_SCHEMA.my_pretrained_models_stage',
   overwrite=True,
   auto_compress=False
)


@cachetools.cached(cache={})
def read_model():
   import_dir = sys._xoptions.get("snowflake_import_directory")
   if import_dir:
       # Load the model
       return joblib.load(f'{import_dir}/summarization.joblib')


@pandas_udf(  
       name='ML_DEMO_DB.ML_DEMO_SCHEMA.get_text_summary',
       session=session,
       is_permanent=True,
       replace=True,
       imports=[
           f'@ML_DEMO_DB.ML_DEMO_SCHEMA.my_pretrained_models_stage/summarization.joblib'
       ],
       input_types=[PandasSeriesType(StringType())],
       return_type=PandasSeriesType(StringType()),
       stage_location='@ML_DEMO_DB.ML_DEMO_SCHEMA.my_pretrained_models_stage',
       packages=['cachetools==4.2.2', 'transformers==4.14.1']
   )

def get_text_summary(texts):   
   # Load the sentiment analysis model from stage
   # using the caching mechanism
   summarization_model = read_model()

    # Apply the model
   summaries = []
   for text in texts:
       result = summarization_model(text)
       summaries.append(result)
   return pd.Series(summaries)


