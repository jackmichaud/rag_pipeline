import environment_variables #environment variables configured in other file

from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

numThemes = 4
numStoriesPerTheme = 1

#llm = Ollama(model="llama3")
llm = ChatOpenAI(temperature=0)

themes_prompt = ChatPromptTemplate.from_template("Return {numberOfThemes} unique short story themes. Separate these themes by just a new line (do not number them or put a bullet)")
output_parser = StrOutputParser()

themes_generation_chain = (
    themes_prompt 
    | llm 
    | output_parser 
    | (lambda x: x.split("\n"))
)

themes = themes_generation_chain.invoke({"numberOfThemes": numThemes}) #generate list of themes

# print("THEMES: \n" + themes)

short_story_prompt = ChatPromptTemplate.from_template("Write {numberOfShortStories} short stories given the following theme: {theme}")

short_story_chain = (
    short_story_prompt
    | llm
    | output_parser
)

stories = []

for t in themes:
    story = short_story_chain.invoke({"numberOfShortStories": numStoriesPerTheme, "theme": t})
    stories.append(story)

print(stories)

#split the stories
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50
)
splits = text_splitter.split_documents(stories)


#embed stories into chroma db
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever() #document retriever






