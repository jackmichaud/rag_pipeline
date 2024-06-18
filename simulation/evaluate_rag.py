import environment_variables
import textwrap

from langsmith.evaluation import LangChainStringEvaluator, evaluate
from langchain_community.llms.ollama import Ollama

from rag_chain import predict_rag_answer, predict_rag_answer_with_context


evaluation_llm = Ollama(model="llama3", temperature=0)

# Evaluators

# Compares RAG answer to ground truth
qa_evalulator = LangChainStringEvaluator(
    "cot_qa",
    prepare_data=lambda run, example: {
        "prediction": run.outputs["answer"],
        "reference": example.outputs["answer"],
        "input": example.inputs["question"],
    },
    config={
        "llm": evaluation_llm
    }
)

# Compares RAG answer to context documents
answer_hallucination_evaluator = LangChainStringEvaluator(
    "labeled_score_string",
    config={
        "criteria": {
            "accuracy": """Is the Assistant's Answer grounded in the Ground Truth documentation? A score of [[1]] means that the
            Assistant answer contains is not at all based upon / grounded in the Groun Truth documentation. A score of [[5]] means 
            that the Assistant answer contains some information (e.g., a hallucination) that is not captured in the Ground Truth 
            documentation. A score of [[10]] means that the Assistant answer is fully based upon the in the Ground Truth documentation."""
        },
        # If you want the score to be saved on a scale from 0 to 1
        "normalize_by": 10,
        "llm": evaluation_llm
    },
    prepare_data=lambda run, example: {
        "prediction": run.outputs["answer"],
        "reference": run.outputs["contexts"],
        "input": example.inputs["question"],
    }
)

# Compares retrieved documents to original question
docs_relevance_evaluator = LangChainStringEvaluator(
    "score_string",
    config={
        "criteria": {
            "document_relevance": textwrap.dedent(
                """The response is a set of documents retrieved from a vectorstore. The input is a question
            used for retrieval. You will score whether the Assistant's response (retrieved docs) is relevant to the Ground Truth 
            question. A score of [[1]] means that none of the  Assistant's response documents contain information useful in answering or addressing the user's input.
            A score of [[5]] means that the Assistant answer contains some relevant documents that can at least partially answer the user's question or input. 
            A score of [[10]] means that the user input can be fully answered using the content in the first retrieved doc(s)."""
            )
        },
        # If you want the score to be saved on a scale from 0 to 1
        "normalize_by": 10,
        "llm": evaluation_llm
    },
    prepare_data=lambda run, example: {
        "prediction": run.outputs["contexts"],
        "input": example.inputs["question"],
    }
)

dataset_name = "RAG_test_catan"
experiment_results = evaluate(
    predict_rag_answer_with_context,
    data=dataset_name,
    evaluators=[qa_evalulator, answer_hallucination_evaluator, docs_relevance_evaluator],
    experiment_prefix="rag-qa-catan",
    metadata={"variant": "Catan context, llama3"},
)
print("Done!")