# Análise de sentimentos: VADER vs RoBERTa 

## Introdução

Este projeto consiste na análise de sentimentos de um conjunto de tweets para companhias aéreas escritos por clientes. Porém, serão utilizadas duas abordagens diferentes, que serão comparadas entre si: VADER (Valence Aware Dictionary and sEntiment Reasoner) e RoBERTa (Robustly Optimized BERT Pretraining Approach). O projeto foi feito inteiramente em Python no Google Colab.

### Identificação 
* Membro: Luiz Eduardo Fernandes Lobato. Matrícula: 20220030040  

### Informações Gerais 
* O objetivo deste projeto é analisar um grande corpo de textos para extrair e extrair os sentimentos presentes nele (negatividade, positividade ou indiferença).
* Foram usados duas abordagens: VADER e RoBERTa.
* O objetivo foi comparar o desempenho de cada abordagem de forma clara. 
* A base de dados contém tweets escritos para companhias aéreas. Ela contém informações como data do tweet, quantidade de retweets, conteúdo textual, seu sentimento rotulado, dentre outros.

## Metodologia 
* O projeto usa bibliotecas e conceitos fundamentais da área de NLP (Natural Language Processing). 
* Não houve treinamento e teste. Ambos os algoritmos utilizam bibliotecas e funções prontas. Eles classificam o texto recebido com parâmetros numéricos que variam de -1 a 1. 
* Os atributos "text" e "airline_sentiment" foram selecionados, pois são essenciais para o objetivo do projeto. 

## Códigos
* Bibliotecas usadas: pandas, numpy, scipy, nltk, transformers, matplotlib, seaborn e tqmd. 
* Encontrando parâmetros VADER para cada tweet:
```
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['text']
    myid = row['tweet_id']
    res[myid] = sia.polarity_scores(text)
```
*  Definindo uma função para calcular parâmetros RoBERTa de um texto:
```
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict
```
* Criando um dataframe com parâmetros VADER e RoBERTa:
```
res_roberta = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['text']
        myid = row['tweet_id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res_roberta[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')

results_df = pd.DataFrame(res_roberta).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})

results_df['sentimentos'] = df['airline_sentiment'].values
results_df.head()
```
* O link para o notebook está [aqui](https://colab.research.google.com/drive/1_rKfdBshYjCvM9nrSWUJxLr4Xgiq35rz?usp=sharing).

## Experimentos 
* Foi feita uma análise visual das abordagens usando gráficos "boxplot" e "barplot". Ao fim, foi usado "pairplot" para uma visão geral.
* Conclusão: RoBERTa teve um desempenho muito mais condizente com os rótulos das base de dados. VADER, embora tenha recursos à primeira vista avançados (sensibilidade a emojis textuais e a pontuação múltipla), teve desempenho ruim, principalmente em textos rotulados 'neutros'; o lado positivo é que ele conseguiu capturar a tendência da resposta correta (um exemplo disso é o "boxplot" dos três parâmetros).

## Conclusão 
* O projeto conseguiu mostrar a diferença de precisão entre as duas abordagens. Além disso, usou uma base de dados extremamente útil e confiável: um conjunto de tweets enviados por pessoas comuns para opinarem sobre empresas, eventos, pessoas.
* Por isso, a análise de sentimentos é uma forma mais natural de fazer uma pesquisa de mercado, que pode substituir entrevistas ou formulários, por exemplo.