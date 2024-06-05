import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from operator import itemgetter

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