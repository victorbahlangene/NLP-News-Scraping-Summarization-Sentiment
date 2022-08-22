#!/usr/bin/env python
# coding: utf-8

# # 1. Install and Import Baseline Dependencies

# In[ ]:


get_ipython().system('pip install transformers')


# In[2]:


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

from bs4 import BeautifulSoup
import requests


# In[ ]:





# # 2. Setup Summarization Model

# In[3]:


model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# In[ ]:





# # 4. Building a News and Sentiment Pipeline

# In[9]:


# search options #
monitored_tickers = []
n = input("Please enter a Data Science topic of interest: ")
monitored_tickers.append(n)


# In[10]:


monitored_tickers


# ## 4.1. Search for Data science News using Medium website

# In[11]:


def search_meduim_urls(monitored_tickers):
    search_url = "https://medium.com/tag/{}".format(monitored_tickers)
    r = requests.get(search_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    # location where link to news is found(a tag with attribute "aria-label"= "Post Preview Title") #
    atags = soup.find_all('a', attrs={"aria-label": "Post Preview Title"})
    hrefs = ['https://medium.com'+link['href'] for link in atags]
    return hrefs 


# In[12]:


# make a dictionary {framework: link_to_article about the framework} #
raw_urls = {framework:search_meduim_urls(framework) for framework in monitored_tickers}
raw_urls


# ## 4.2. Strip out unwanted URLs

# In[ ]:


# not nessesary here #


# In[13]:


cleaned_urls = raw_urls


# In[14]:


cleaned_urls


# ## 4.3. Search and Scrape Cleaned URLs

# In[15]:


def scrape_and_process(URLs):
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


# In[16]:


articles = {ticker:scrape_and_process(cleaned_urls[ticker]) for ticker in monitored_tickers}
articles


# In[18]:


articles["fastai"][2]


# In[19]:


articles["fastai"][5]


# ## 4.4. Summarise all Articles

# In[20]:


def summarize(articles):
    summaries = []
    for article in articles:
        input_ids = tokenizer.encode(article, return_tensors='pt',max_length=512, truncation=True)
        output = model.generate(input_ids, max_length=56, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries


# In[21]:


# takes 3 mins to execute #
summaries = {ticker:summarize(articles[ticker]) for ticker in monitored_tickers}
summaries


# In[22]:


summaries["fastai"]


# # 5. Adding Sentiment Analysis

# In[23]:


# using pipeline #
sentiment = pipeline('sentiment-analysis')


# In[24]:


# test sentmrnt analysis with 'summaries["fastai"]' #
sentiment(summaries["fastai"])


# In[25]:


scores = {ticker:sentiment(summaries[ticker]) for ticker in monitored_tickers}
scores


# In[26]:


print(summaries['fastai'][3], scores['fastai'][3]['label'], scores['fastai'][3]['score'])


# sentiment analysis can be improved by finetuning to a datascience specific dataset.

# In[ ]:





# # 6. Exporting Results to CSV

# In[27]:


summaries


# In[28]:


scores


# In[29]:


cleaned_urls


# In[30]:


range(len(summaries['fastai']))


# In[31]:


summaries['fastai'][3]


# In[32]:


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


# In[33]:


final_output = create_output_array(summaries, scores, cleaned_urls)
final_output


# In[34]:


# adding cols #
final_output.insert(0, ['Ticker', 'Summary', 'Label', 'Confidence', 'URL']) 


# In[35]:


final_output


# **Export results**

# In[ ]:


import csv
with open('assetsummaries.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)

