# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script TCC - NPL - POLITICA .

"""
import pandas as pd                                  ## carregar bases 
import string                                        ## pre processamento 
##import random                                      ## gerar randoms
import seaborn as sns                                ## geração de grafics
import numpy as py                                   ## lib cietifica     
import spacy
import nltk
import re 
import matplotlib.pyplot as plt
import base64
from io import BytesIO

pln_pt = spacy.load('pt_core_news_sm')               ## carrega estrutura PT 
nltk.download('rslp')                                ## LIB npl em portugues 
####stampt = nltk.stem.RSLPStemmer()                     ## STAMIN RAIZ PALAVRA 
from spacy.lang.pt.stop_words import STOP_WORDS      ## STOP WORDS PT  416 palavras 
from wordcloud import WordCloud, ImageColorGenerator  ## WordCloud
from collections import Counter 

## pre processamento 
stop_words = STOP_WORDS
counter = Counter

##  processamento analise de sentimento 
def pre_processamento(x):
    ## caixa baixa
    x = x.lower()
    ## retira @
    x = re.sub(r"@[A-Za-z0-9$-_@.&]+",' ',x) 
    ## links
    x= re.sub(r"^https?://\/\/.*[\r\n]*[A-Za-z0-9./]+",' ',x)
    ## espaco em branco 
    x= re.sub(r" +",' ',x)
    ## retira /n 
    x= re.sub(r"\n+",' ',x)
    ## lematização 
    regrex_pattern=re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                           "]+", flags = re.UNICODE)
    x=regrex_pattern.sub(r'',x)
    
   
    
    doc = pln_pt(x)
    lista =[]
    for token in doc :
        lista.append(token.lemma_)
        
    ## stopwords 
    lista = [palavra for palavra in lista if palavra not in stop_words 
             and palavra not in string.punctuation
             and palavra not in (':)',':p',':P',':d',':D',':(' )]    
    ## tranforma lista em string (fomato frase novamente )
    lista = ' '.join([str(elemento)for elemento in lista if not elemento.isdigit()])    

    return lista 

## textminer 
def processamentominer(x):
    ## caixa baixa
    x = x.lower()
    ## retira @
    x = re.sub(r"@[A-Za-z0-9$-_@.&... ]+",' ',x) 
    ## links
    x= re.sub(r"https?://[A-Za-z0-9./]+",' ',x)
    ## espaco em branco 
    x= re.sub(r" +",' ',x)
    ## retira /n 
    x= re.sub(r"\n+",' ',x)
    ## lematização 
    x= re.sub(r"^https?://\/\/.*[\r\n]*[A-Za-z0-9./]+",' ',x)
    ## espaco em branco 
    x= re.sub(r" +",' ',x)
    ## retira /n 
    x= re.sub(r"\n+",' ',x)
    ## lematização 
    regrex_pattern=re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                           "]+", flags = re.UNICODE)
    x=regrex_pattern.sub(r'',x)
    
    doc = pln_pt(x)
    lista =[]
    for token in doc :
        lista.append(token.lemma_)
        
    ## stopwords 
    x = [palavra for palavra in lista if palavra not in stop_words and 
         palavra not in string.punctuation and  palavra.isdigit()==False and 
         palavra not in (':)',':p',':P',':d',':D',':(' ) and 
         palavra !='...' and palavra not in('name','length','dtype','object',' ','parir')]    
    ## tranforma lista em string (fomato frase novamente )
    ## lista = ' '.join([str(elemento)for elemento in lista if not elemento.isdigit()])    
    
    return x 

##resultado = pre_processamento(base_teste)
## identificando principais assuntos atravez de entidades 

def returnentidade (x):
    var_ent = ['LOC','PER','ORG'] ## LOC = Localizacao , PER = Pessoas , Organizacao 
    var_doc = pln_pt(str(x))
    lista_ent = []
    for entidade  in var_doc.ents : 
         if entidade.label_ in var_ent :
             lista_ent.append(str(entidade.text))
         else :lista_ent.append(str('Não foi encontrado similaridades de palavras'))
        
   ##return lista_ent.append(entidade.text) 
    ##if entidade.label_ == 'MISC':
    ## print (entidade.text,entidade.label_)
    return  lista_ent   


def analise_texto(texto_site):

    base_tlix = texto_site ## base input
    
    
    ## base treinamento   
    base_polit_train = pd.read_csv('data/DS_POLITIZE_TCC.csv') ## arquivo com o modelo
    ## base teste 
    p1 = pln_pt(str(processamentominer(str(base_polit_train))))
    p2 = pln_pt(str(processamentominer(str(base_tlix))))

    # similaridade
    resultado = round((p1.similarity(p2))*100,2)

    ## wordcloud
    palavras_word = pln_pt(str(returnentidade(base_tlix)))

    wordcloud = WordCloud(stopwords=stop_words,
                            background_color='white', width=800,                            
                            height=400).generate(str(palavras_word))
    fig, ax = plt.subplots(figsize=(8,4))            
    ax.imshow(wordcloud, interpolation='bilinear')       
    ax.set_axis_off()
    plt.imshow(wordcloud)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    html_wordcloud = f"<img src='data:image/png;base64,{data}' width='200' height='100'/>"

    ### frequencia palavra 
    freq_texto = processamentominer(str(base_tlix))
    df = pd.DataFrame(freq_texto)
    df_stack=df.stack(level=0) 
    counter=df_stack.value_counts()[0:10] # top 10 palavras com maior frequencia 
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(counter.index,counter.values) ## palavra e qtd   

    buf2 = BytesIO()
    fig.savefig(buf2, format="png")
    data2 = base64.b64encode(buf2.getbuffer()).decode("ascii")

    html_freq = f"<img src='data:image/png;base64,{data2}'/>"

    return resultado, html_wordcloud, html_freq

    

    ## variaveis saida site ##
    # 1. RESULTADO 
    # 2. WORDCLOUD
    # 3. COUNTER 
    ##################################################### fim ###################################
    ## analise de texto 
