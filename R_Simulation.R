library(reticulate)

# Ensure use of proper python env
use_python("anaconda3/bin/python")

# Load python scripts
# This can be used to do data analysis
source_python(paste(dirname(rstudioapi::getSourceEditorContext()$path),"/data_generation.py",sep=""))
source_python(paste(dirname(rstudioapi::getSourceEditorContext()$path),"/rag_pipeline.py",sep=""))



query_rag("Which resource cards are there in Catan?")