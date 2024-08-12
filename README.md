# rag_pipeline

To set up the app, run the following commands:

```sh
cat <<EOL > environment_variables.py
import os

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = YOUR_API_KEY
os.environ['OPENAI_API_KEY'] = YOUR_API_KEY

# Silence warnings from HuggingFace
os.environ["TOKENIZERS_PARALLELISM"] = "false"
EOL
```  

Once the environment is set up, to launch the streamlit app, run the following command:

```sh
streamlit run app/app.py
```  

Useful Sources:  
https://www.youtube.com/watch?v=2TJxpyO3ei4  
https://github.com/pixegami/rag-tutorial-v2  
https://docs.smith.langchain.com/tutorials/Developers/rag  
https://stackoverflow.blog/2023/10/18/retrieval-augmented-generation-keeping-llms-relevant-and-current/  
https://www.youtube.com/watch?v=8OJC21T2SL4  
https://www.youtube.com/watch?v=bjb_EMsTDKI&list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&index=2  

