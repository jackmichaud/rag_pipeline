library(reticulate)

# Ensure use of proper python env
use_python("anaconda3/bin/python")

# Load python scripts
# This can be used to do data analysis
source_python(paste(dirname(rstudioapi::getSourceEditorContext()$path),"/data_generation.py",sep=""))
source_python(paste(dirname(rstudioapi::getSourceEditorContext()$path),"/rag_pipeline.py",sep=""))

llm <- define_llm(temperature = 1, top_k = 8, top_p = 0.5)

themes <- generate_themes_list(llm, num_themes = 4)

generate_stories(llm, themes = themes, num_storeies_per_theme = 1)
