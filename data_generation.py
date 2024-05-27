import environment_variables #environment variables configured in other file

from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

#llm = ChatOpenAI(temperature=0)

# Wrapper function to define llm from R
def define_llm(temperature: float, top_k: float, top_p: float):
    llm = Ollama(model="llama2", temperature=temperature, top_k=top_k, top_p=top_p)
    return llm

# Generate a list of themes given an llm object and a number of themes
def generate_themes_list(llm, num_themes: int):
    themes_prompt = ChatPromptTemplate.from_template("Return soleley {numberOfThemes} unique short story themes. Separate these themes by just a new line (do not number them or put a bullet)")

    # Generate themes chain
    themes_generation_chain = (
        themes_prompt 
        | llm 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )
    themes = themes_generation_chain.invoke({"numberOfThemes": num_themes}) #generate list of themes

    for t in themes:
        if t == "":
            themes = themes[1:]

    #print("THEMES: \n" + themes)
    return themes

# Generate stories given an llm and a list of themes
def generate_stories(llm, themes: list, num_storeies_per_theme: int):
    short_story_prompt = ChatPromptTemplate.from_template("Write {numberOfShortStories} short stories given the following theme: {theme}")

    short_story_chain = (
        short_story_prompt
        | llm
        | StrOutputParser()
    )

    stories = []

    # Call the short story chain for each theme and add it to the list of stories
    for t in themes:
        story = short_story_chain.invoke({"numberOfShortStories": num_storeies_per_theme, "theme": t})
        stories.append(story)

    print(stories)

def load_stories(stories: list):
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