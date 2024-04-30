from langchain_openai import ChatOpenAI
import toml
# Configuration


# old api-key "sk-proj-ycb9fbckOogkeyU3WSNdT3BlbkFJ876crdzfoKdEoMJh10gr"
# sk-proj-DNmVO6nqaLu328iKnxouT3BlbkFJrPnKXVQwIqGELHYCGszX
config = toml.load("config.toml")
api_key = config['API_KEYS']['OPENAI']
# # Initialize ChatOpenAI
# llm = ChatOpenAI(openai_api_key=api_key)
llm = ChatOpenAI(model_name='gpt-4-turbo', temperature=0.1, openai_api_key=api_key)
# 