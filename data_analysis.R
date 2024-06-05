library(reticulate)
# Ensure use of proper python env
use_python("anaconda3/bin/python")
# Load python scripts
# This can be used to do data analysis
source_python(paste(dirname(rstudioapi::getSourceEditorContext()$path),"/data_generation.py",sep=""))
source_python(paste(dirname(rstudioapi::getSourceEditorContext()$path),"/rag_pipeline.py",sep=""))
source_python(paste(dirname(rstudioapi::getSourceEditorContext()$path),"/simulation_code.py",sep=""))
# This llm instance is more creative and is used to generate unique themes and stories
llm <- define_llm(temperature = 1, top_k = 8, top_p = 0.5)
themes <- generate_themes_list(llm, num_themes = 4)
themes
stories <- generate_stories(llm, themes = themes)
stories
llm <- define_llm(temperature = 0, top_k = 8, top_p = 0.5)
sum <- 0
index <- 1
for(story in stories) {
  indentified_theme <- identify_story_theme(llm, story = story)  
  similarity <- cosine_similarity(themes[[index]], indentified_theme)
  print(paste(indentified_theme, themes[[index]], similarity, sep=" : "))
  sum <- sum + similarity
  index <- index + 1
}
average_accuracy = sum / (index - 1)
average_accuracy


