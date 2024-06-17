import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from operator import itemgetter
from random_word import RandomWords
import nltk 
from nltk.corpus import wordnet 

from get_embedding_function import get_embedding_function

import numpy as np
from numpy.linalg import norm

def identify_story_theme(llm, story: str):
    prompt_template = """Identify the theme of the following short story: {story} \n---\n Identify the theme of the above story. Do not explain why, just list the theme. Put for newlines exactly before listing the theme"""
    hyde_prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt_string = hyde_prompt.format(story=story)

    response = llm.invoke(prompt_string)

    # Remove the part of the string before the fourth newline character
    parts = response.split("\n\n\n\n")
    response = parts[-1] if len(parts) > 1 else response

    return response


def cosine_similarity(a, b):
    embedding_function = get_embedding_function()
    embedding_a = embedding_function.embed_query(a)
    embedding_b = embedding_function.embed_query(b)
    A = np.array(embedding_a)
    B = np.array(embedding_b)
    similarity = np.dot(A, B)/(norm(A)*norm(B))
    return similarity

def semantic_similarity(a, b):
    return wordnet.synsets(a)[0].wup_similarity(wordnet.synsets(b)[0])

def llm_similarity(a, b):
    prompt_template = """Identify the similarity between the following phrases on a scale from 1-10: {a} and {b}. Simly output a number, nothing else"""
    hyde_prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt_string = hyde_prompt.format(a=a, b=b)
    llm = Ollama(model="llama3", temperature="0")
    response = llm.invoke(prompt_string)

    return response

def generate_random_word():
    r = RandomWords()
    return r.get_random_word()

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonyms.append(l.name()) 
    
    return list(set(synonyms))


def test_similarity_of_synonyms(word):
    synonyms = get_synonyms(word)
    if(len(synonyms) <= 3): 
        return
    sum = 0
    for synonym in synonyms:
        similarity = cosine_similarity(word, synonym)
        #similarity = wordnet.synsets(word)[0].wup_similarity(wordnet.synsets(synonym)[0])
        print(word, synonym, similarity)
        sum += similarity
    
    print("Average similarity: ", sum/len(synonyms))
    return sum/len(synonyms)

def benchmark_embedding_similarity(num_simulations):
    sum_similarity = 0
    for i in range(num_simulations):
        similarity = test_similarity_of_synonyms(generate_random_word())
        while not isinstance(similarity, float):
            similarity = test_similarity_of_synonyms(generate_random_word())
        sum_similarity += similarity

    print("Total average similarity: ", sum_similarity/num_simulations)
    return sum_similarity/num_simulations



if __name__ == "__main__":
    benchmark_embedding_similarity(100)
