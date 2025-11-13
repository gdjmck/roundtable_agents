import os
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_BASE = os.getenv('OPENAI_BASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MLLM_BASE_URL = os.getenv('MLLM_BASE_URL')
SF_BASE_URL = os.getenv('SF_BASE_URL')
SF_API_KEY = os.getenv('SF_API_KEY')

llm = ChatOpenAI(model="deepseek_qwen_32b", temperature=0, openai_api_base=OPENAI_API_BASE,
                 openai_api_key=OPENAI_API_KEY)
llm_with_json_mode = llm.bind(response_format={"type": "json_object"})
mllm = ChatOpenAI(model="Qwen/Qwen2.5-VL-7B-Instruct", max_tokens=2048,
                  base_url=MLLM_BASE_URL,
                  temperature=0.1,
                  http_client=httpx.Client(transport=httpx.HTTPTransport(verify=False))
                  )

mllm_sf_235B = ChatOpenAI(model='Qwen/Qwen3-VL-235B-A22B-Thinking',
                          openai_api_base=SF_BASE_URL,
                          max_tokens=2048, temperature=0.1,
                          openai_api_key=SF_API_KEY)

mllm_sf_30B = ChatOpenAI(model='Qwen/Qwen3-VL-30B-A3B-Thinking',
                          openai_api_base=SF_BASE_URL,
                          max_tokens=2048, temperature=0.1,
                          openai_api_key=SF_API_KEY)
