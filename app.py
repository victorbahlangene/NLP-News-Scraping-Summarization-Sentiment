import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

from bs4 import BeautifulSoup
import requests

# app layout #
st.set_page_config(
    page_title="Medium News App"
)

## FUNCTIONS ##

# search medium urls function #
@st.cache_resource
def search_meduim_urls(monitored_tickers):
  search_url = "https://medium.com/tag/{}".format(monitored_tickers)
  r = requests.get(search_url)
  soup = BeautifulSoup(r.text, 'html.parser')
  # location where link to news is found(a tag with attribute "aria-label"= "Post Preview Title") #
  atags = soup.find_all('a', attrs={"aria-label": "Post Preview Title"})
  hrefs = ['https://medium.com'+link['href'] for link in atags]
  return hrefs 

# funtion to search and scrape cleaned urls #
@st.cache_resource
def scrape_and_process(URLs):
  """
  - function grabs all p-tags.
  - create list of whats in every p tag.
  - plit list into individual words, max 350.
  - make 1 corpus of data.
  - the length of each article tokens will be 350,
    because the max of the model i am using is 512 and i want the app to be faster.
  """
  ARTICLES = []
  for url in URLs: 
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = [paragraph.text for paragraph in paragraphs]
    words = ' '.join(text).split(' ')[:350]
    ARTICLE = ' '.join(words)
    ARTICLES.append(ARTICLE)
  return ARTICLES

#function to Summarise all Articles#
@st.cache_resource
def summarize(articles,_tokenizer,_model):
  """
  encode , generate, decode, append to list
  """
  summaries = []
  for article in articles:
    input_ids = _tokenizer.encode(article, return_tensors='pt',max_length=512, truncation=True)
    output = _model.generate(input_ids, max_length=56, num_beams=5, early_stopping=True)
    summary = _tokenizer.decode(output[0], skip_special_tokens=True)
    summaries.append(summary)
  return summaries

# function to load the transformer #
@st.cache_resource
def load_summary_transformer():  
  # load transformers #
  model_name = "facebook/bart-large-cnn"
  tokenizer_summary = AutoTokenizer.from_pretrained(model_name)
  model_summary = AutoModelForSeq2SeqLM.from_pretrained(model_name)

  return tokenizer_summary, model_summary

# function to load sentiment pipeline #
@st.cache_resource
def load_sentiment_pipeline():
  sentiment = pipeline('sentiment-analysis')
  
  return sentiment

# function to create final output #
def create_output_array(summaries, scores, urls):
  output = []
  for ticker in monitored_tickers:
    for counter in range(len(summaries[ticker])):
      output_this = [
          ticker,
          summaries[ticker][counter],
          scores[ticker][counter]['label'],
          scores[ticker][counter]['score'],
          urls[ticker][counter]
      ]
      output.append(output_this)
  return output

# display summary output #
def cards(title,score,sentiment,article,link):
    return f"""
    <div class="card bg-light mb-3">
        <div class="card-body">
            <h5 class="card-title">{title}</h5>
            <h6 class="card-subtitle mb-2 text-muted">The article is: {score*100:.2f}% {sentiment}.</h6>
            <p class="card-text">{article}.</p>
            <a href={link} class="card-link">Link to article</a>
        </div>
    </div>
    <br></br>
    """

# function to load bootstrap #
def boot():
  return """
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
  """

# load bootstrap #
st.markdown(boot(), unsafe_allow_html=True)

# load_summary_transformer #
tokenizer_summary, model_summary = load_summary_transformer()

# load sentiment pipeline #
sentiment = load_sentiment_pipeline()



## APP OUTPUT ##
st.markdown("<h1 style='text-align: center; color: grey;'>Medium News App</h1>",
            unsafe_allow_html=True)

# containers #
col1, col2, col3 = st.columns(3)

# session_state user input initilization #
if 'user_final_input' not in st.session_state:
  st.session_state['user_final_input'] = ''

# SEARCH SECTION #
with st.expander("Make inquiry"):
  st.markdown("<h2 style='text-align: center; color: black;'>Summary</h2>",
              unsafe_allow_html=True)
  # user input #
  monitored_tickers = []

  # user input options #
  option = st.selectbox(
    'Some options to select',
    ('chatgpt', 'fastai', 'pytorch', 'tensorflow',('manual entry'))
    )
  # allows for manual search entry #
  if option=="manual entry":
    user_select = st.text_input(
          "Please enter a Data Science topic of interest: ")
    monitored_tickers.append(user_select)
    st.write(user_select)
    st.session_state['user_final_input'] = user_select
  else:
    monitored_tickers.append(option)
    st.write(option)
    st.session_state['user_final_input'] = option

    

  # how many summaries to inference #
  summary_count = st.slider('How many summaries do you want?', 1, 5, 1)
  st.write("I'm selecting ", summary_count, 'summaries.')
  if summary_count == 3:
    st.markdown(f"""
            <div class="alert alert-warning" role="alert">
              The summary will take about 1 minute to process.
            </div>
    """
    , unsafe_allow_html=True)
  elif summary_count == 4 or summary_count == 5:
    st.markdown(f"""
            <div class="alert alert-danger" role="alert">
              The summary will take about 2 minutes to process.
            </div>
    """
    , unsafe_allow_html=True)

  
          
  with st.form(key="user_input"): 
    summary = st.form_submit_button("Summary")
    if summary:
      # test function #
      search_meduim_urls(monitored_tickers[0])
      # make a dictionary {framework: link_to_article about the framework} #
      cleaned_urls= {framework:search_meduim_urls(framework) for framework in monitored_tickers}

      articles = {ticker:scrape_and_process(cleaned_urls[ticker]) for ticker in monitored_tickers}
      
      articles[st.session_state['user_final_input']] = articles[st.session_state['user_final_input']][:summary_count]
      #articles[option] = articles[option][:summary_count]
      
      #articles

      # summary #
      # 1m 25s to sumarize #
      summaries = {ticker:summarize(articles[ticker],tokenizer_summary, model_summary) for ticker in monitored_tickers}
      
      
      scores = {ticker:sentiment(summaries[ticker]) for ticker in monitored_tickers}
      #scores

      final_output = create_output_array(summaries, scores, cleaned_urls)
      #final_output

      #final_output[0]
      for i in range(len(final_output)):
        st.markdown(
          cards(
              final_output[i][0],
              final_output[i][3],
              final_output[i][2],
              final_output[i][1],
              final_output[i][4]
          ), 
          unsafe_allow_html=True)
        

