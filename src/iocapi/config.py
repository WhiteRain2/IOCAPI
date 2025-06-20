import os
import pathlib
import json
import asyncio
from concurrent.futures import ProcessPoolExecutor
from collections import namedtuple
from typing import Sequence

from dotenv import load_dotenv
from loguru import logger
from apiutils import LLMService, API
from get_top_k_q.get_top_k import get_top_k_apis
from utils import PromptUtils

load_dotenv(override=True)


class PathConfig:
    BASE_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve()
    DATA_DIR: pathlib.Path = BASE_DIR / "data"
    LOG_DIR: pathlib.Path = BASE_DIR / "logs"
    PROMPT_DIR: pathlib.Path = BASE_DIR / "prompts"


class LLMConfig:
    MODEL_NAME: str = ''
    BASE_CONFIG: dict = {
        "temperature": 0.7,
        "seed": 42,
    }
    CLIENT_CONFIG: dict = {
        "base_url": os.getenv("BASE_URL"),
        "api_key": os.getenv("API_KEY"),
    }


class ClarifyConfig(LLMConfig):
    MODEL_NAME = "qwen-max-latest"
    BASE_CONFIG = {
        "temperature": 0.7,
        "seed": 42,
        'response_format': {"type": "json_object"}
    }
    ClarifyResponse = namedtuple("ClarifyResponse",
                                 ["demo_input", "demo_output", "statement", "tokens"])

    @classmethod
    async def clarifies(cls,
                        queries: Sequence[str],
                        tqdm_title='Clarifying') -> list["ClarifyConfig.ClarifyResponse"]:
        prompt_path = PathConfig.PROMPT_DIR / "clarifier.md"
        prompt = PromptUtils(prompt_path)
        clarifier = LLMService(cls.MODEL_NAME,
                               prompt.sys_prompt,
                               cls.BASE_CONFIG)
        qs = [prompt.get_prompt(query=q) for q in queries]
        results = await clarifier.queries(qs, tqdm_title=tqdm_title)
        return await asyncio.gather(*[cls.parse_answer(r.answer, r.tokens) for r in results])

    @classmethod
    async def parse_answer(cls, response: str, tokens: int) -> "ClarifyConfig.ClarifyResponse":
        try:
            ans = json.loads(response)
            demo_input = ans.get("input", None)
            demo_output = ans.get("output", None)
            statement = ans.get("statement", None)
            if demo_input and demo_output and statement:
                return cls.ClarifyResponse(
                    demo_input=demo_input,
                    demo_output=demo_output,
                    statement=statement,
                    tokens=tokens
                )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response: {e}")
        return cls.ClarifyResponse(None, None, None, 0)

    @classmethod
    def get_similar_apis(cls,
                         statement: str,
                         top_k: int) -> list[API]:
        raw_apis = get_top_k_apis(statement, top_k)
        similar_apis = [API(api) for api in raw_apis]

        standard_apis = API.get_standard_apis()
        for i in range(len(similar_apis)-1, -1, -1):
            api = similar_apis[i]
            if api.is_standard:
                for standard_api in standard_apis:
                    if api.fullname == standard_api.fullname:
                        similar_apis[i] = standard_api
                        break
            else:
                doc_apis = api.get_possible_standard_apis(first=True)
                if doc_apis:
                    similar_apis[i] = doc_apis[0]
                else:
                    similar_apis.pop(i)

        return similar_apis

    @classmethod
    async def batch_get_similar_apis(cls,
                                     statements: Sequence[str],
                                     top_k: int,
                                     max_workers: int | None = (os.cpu_count() or 0)//2
                                     ) -> list[list[API]]:
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            tasks = [
                loop.run_in_executor(pool, cls.get_similar_apis, stmt, top_k)
                for stmt in statements
            ]

            results: list[list[API]] = await asyncio.gather(*tasks)
        return results


class CoderConfig(LLMConfig):
    MODEL_NAME = "qwen-max-latest"
    TOP_K: int = 8
    CodeResponse = namedtuple("CodeResponse",
                              ["code", "apis", "add_info", "tokens"])

    @classmethod
    async def code(cls,
                   clarifiers_res: Sequence[ClarifyConfig.ClarifyResponse],
                   java_api_lists: Sequence[Sequence[API]]) -> list["CoderConfig.CodeResponse"]:
        prompt_path = PathConfig.PROMPT_DIR / "coder.md"
        prompt = PromptUtils(prompt_path)
        coder = LLMService(CoderConfig.MODEL_NAME,
                           prompt.sys_prompt,
                           CoderConfig.BASE_CONFIG)
        qs = await asyncio.gather(*[cls.construct_prompt(prompt, c_res, java_api_list)
                                  for c_res, java_api_list
                                  in zip(clarifiers_res, java_api_lists)])
        results = await coder.queries(qs, tqdm_title='Coding')
        answers = await asyncio.gather(*[cls.parse_coder_response(r.answer) for r in results])
        return [CoderConfig.CodeResponse(
                code=ans[0],
                apis=ans[1],
                add_info=ans[2],
                tokens=r.tokens
                ) for ans, r in zip(answers, results)]

    @classmethod
    async def construct_prompt(cls,
                               prompt: PromptUtils,
                               clarifier_res: ClarifyConfig.ClarifyResponse,
                               java_api_list: Sequence[API]):
        api_list_format = ""
        for i, api in enumerate(java_api_list):
            api_list_format += f"API {i + 1}: {api.fullname}\n"
            api_list_format += f"Description: {api.description}\n\n"
        return prompt.get_prompt(query=clarifier_res.statement,
                                 demo_input=clarifier_res.demo_input,
                                 demo_output=clarifier_res.demo_output,
                                 java_api_list=api_list_format)

    @classmethod
    async def parse_coder_response(cls, response: str) -> tuple[str, list[API], str]:
        info = [ch for ch in response.split('#') if ch]
        code = '-'
        apis = []
        add_info = response
        for chunk in info:
            if 'Java Code' in chunk:
                # Parse ```java ``` code block
                try:
                    code = chunk.split('Java Code')[1].split('```java')[1].split('```')[0].strip()
                except Exception as e:
                    logger.error(f"Parse Error {e}: {chunk}")
                    code = str(chunk)
            elif 'API Used' in chunk:
                apis = API.from_string(str(chunk))
            elif 'Additional explanation' in chunk:
                add_info = chunk.split('\n')[1].strip()
        return code, apis, add_info


async def test():
    LLMService.set_llm_client_config(**LLMConfig.CLIENT_CONFIG)
    # Tests
    queries = ["How to parse a JSON string in Java?",
               "What is the best way to sort a list?",
               "How to read a file in Java?"]
    clarifier_res = await ClarifyConfig.clarifies(queries)
    print(clarifier_res[0])
    apis = await ClarifyConfig.batch_get_similar_apis([c.statement for c in clarifier_res],
                                                      CoderConfig.TOP_K)
    code_res = await CoderConfig.code(clarifier_res, apis)
    print(code_res[0])

if __name__ == "__main__":
    asyncio.run(test())
