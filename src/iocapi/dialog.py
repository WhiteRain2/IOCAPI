import json
import asyncio
from typing import Sequence

from loguru import logger
from get_top_k_q.get_top_k import get_top_k_apis
from config import PathConfig, ClarifyConfig, CoderConfig, LLMConfig
from utils import PromptUtils
from apiutils import LLMService, API

LLMService.set_llm_client_config(**LLMConfig.CLIENT_CONFIG)


async def get_similar_apis(statement: str,
                           top_k: int = CoderConfig.TOP_K) -> list[API]:
    similar_apis = [API(api) for api in get_top_k_apis(statement, top_k)]

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


async def code(clarifier_res: ClarifyConfig.ClarifyResponse,
               java_api_list: Sequence[API]) -> CoderConfig.CodeResponse:
    prompt_path = PathConfig.PROMPT_DIR / "coder.md"
    prompt = PromptUtils(prompt_path)
    coder = LLMService(CoderConfig.MODEL_NAME,
                       prompt.sys_prompt,
                       CoderConfig.BASE_CONFIG)
    api_list_format = ""
    for i, api in enumerate(java_api_list):
        api_list_format += f"API {i + 1}: {api.fullname}\n"
        api_list_format += f"Description: {api.description}\n\n"
    q = prompt.get_prompt(query=clarifier_res.statement,
                          demo_input=clarifier_res.demo_input,
                          demo_output=clarifier_res.demo_output,
                          java_api_list=api_list_format)
    response, token = await coder.query(q)
    code, apis, add_info = await CoderConfig.parse_coder_response(response)
    return CoderConfig.CodeResponse(
        code=code,
        apis=apis,
        add_info=add_info,
        tokens=token
    )


async def clarify(query: str) -> ClarifyConfig.ClarifyResponse:
    prompt_path = PathConfig.PROMPT_DIR / "clarifier.md"
    prompt = PromptUtils(prompt_path)
    clarifier = LLMService(ClarifyConfig.MODEL_NAME,
                           prompt.sys_prompt,
                           ClarifyConfig.BASE_CONFIG)
    q = prompt.get_prompt(query=query)
    tokens, retry = 0, 0
    demo_input, demo_output, statement = None, None, None
    ans = {}
    while retry < 3:
        res, t = await clarifier.query(q)
        tokens += t
        try:
            ans = json.loads(res)
            demo_input = ans.get("input", None)
            demo_output = ans.get("output", None)
            statement = ans.get("statement", None)
            if demo_input and demo_output and statement:
                break
            else:
                raise AttributeError("Invalid response structure")
        except json.JSONDecodeError:
            retry += 1
            logger.warning(f"Clarifier response is not JSON, retrying... {retry}")
        except AttributeError:
            retry += 1
            logger.warning(f"Clarifier response is missing required fields, retrying... {retry}")
    else:
        logger.error("Clarifier failed to return valid JSON after 3 retries")
    return ClarifyConfig.ClarifyResponse(
        demo_input=demo_input,
        demo_output=demo_output,
        statement=statement,
        tokens=tokens
    )


async def dialog():
    q = input("Question: ")
    clarifier_res = await clarify(q)
    while True:
        print("You Means:")
        print(f"Input: {clarifier_res.demo_input}")
        print(f"Output: {clarifier_res.demo_output}")
        print(f"You want to ask: {clarifier_res.statement}")
        res = input("User response (yes or what you want): ")
        if res.lower() in ['y', 'yes']:
            break
        clarifier_res = await clarify(q + '\n' + res)

    similar_apis = await get_similar_apis(clarifier_res.statement)
    coder_res = await code(clarifier_res, similar_apis)
    print("Demonstrate Code:")
    print(coder_res.code)
    print("APIs You Might Need:")
    standard_apis = API.get_standard_apis()
    for api in coder_res.apis:
        for a in standard_apis:
            if a.fullname == api.fullname:
                print(f" - {a.fullname}: {a.description}")
                break
    print(f"Tokens Used: {clarifier_res.tokens + coder_res.tokens}")


if __name__ == "__main__":
    asyncio.run(dialog())
