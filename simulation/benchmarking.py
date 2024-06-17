import environment_variables
import uuid

from tqdm import tqdm

from operator import itemgetter
from typing import Sequence

from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable.passthrough import RunnableAssign
from langsmith import traceable

from populate_vectorstore import langchain_docs, retriever

# Generate a unique run ID for this experiment
run_uid = uuid.uuid4().hex[:6]

# After the retriever fetches documents, this
# function formats them in a string to present for the LLM
def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = (
            f"<document index='{i}'>\n"
            f"<source>{doc.metadata.get('source')}</source>\n"
            f"<doc_content>{doc.page_content}</doc_content>\n"
            "</document>"
        )
        formatted_docs.append(doc_string)
    formatted_str = "\n".join(formatted_docs)
    return f"<documents>\n{formatted_str}\n</documents>"


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant answering the question {question}."
            "\n{context}\n"
            "Respond solely based on the document content.",
        ),
        ("human", "{question}"),
    ]
)
llm = Ollama(model="llama2", temperature=1)

response_generator = (prompt | llm | StrOutputParser()).with_config(
    run_name="GenerateResponse",
)

# This is the final response chain.
# It fetches the "question" key from the input dict,
# passes it to the retriever, then formats as a string.

chain = (
    RunnableAssign(
        {
            "context": (itemgetter("question") | retriever | format_docs).with_config(
                run_name="FormatDocs"
            )
        }
    )
    # The "RunnableAssign" above returns a dict with keys
    # question (from the original input) and
    # context: the string-formatted docs.
    # This is passed to the response_generator above
    | response_generator
)

#print(chain.invoke({"question": "What's expression language?"}))

from langchain import hub
from langchain_openai import ChatOpenAI




# Grade prompt
grade_prompt_answer_accuracy = prompt = hub.pull("langchain-ai/rag-answer-vs-reference")

def answer_evaluator(run, example) -> dict:
    """
    A simple evaluator for RAG answer accuracy
    """

    # Get question, ground truth answer, RAG chain answer
    input_question = example.inputs["input_question"]
    reference = example.outputs["output_answer"]
    prediction = run.outputs["answer"]

    # LLM grader
    llm = Ollama(model="llama2", temperature=0)

    # Structured prompt
    answer_grader = grade_prompt_answer_accuracy | llm

    # Run evaluator
    score = answer_grader.invoke({"question": input_question,
                                  "correct_answer": reference,
                                  "student_answer": prediction})
    score = score["Score"]

    return {"key": "answer_v_reference_score", "score": score}

from langsmith.evaluation import evaluate

def predict_rag_answer(example: dict):
    """Use this for answer evaluation"""
    response = rag_bot.get_answer(example["input_question"])
    return {"answer": response["answer"]}




exit(0)
experiment_results = evaluate(
    predict_rag_answer,
    data="catan_dataset",
    evaluators=[answer_evaluator],
    client=client,
    experiment_prefix="rag-answer-v-reference",
    metadata={"version": "catan context, llama2"},
)