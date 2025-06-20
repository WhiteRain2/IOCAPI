import asyncio

import pandas as pd
from apiutils import dataset as dt
from apiutils import Calculator, LLMService
from config import PathConfig, ClarifyConfig, CoderConfig, LLMConfig

LLMService.set_llm_client_config(**LLMConfig.CLIENT_CONFIG)

async def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.index.name = "idx"
    df.to_csv(filename, index=True, encoding="utf-8")


async def batch_clarify(queries, dataset, save_to_file=True):
    clarifier_res = await ClarifyConfig.clarifies(queries)

    similar_apis = await ClarifyConfig.batch_get_similar_apis(
        [res.statement or queries[i] for i, res in enumerate(clarifier_res)],
        top_k=CoderConfig.TOP_K,
    )
    if save_to_file:
        results = []
        for q, c_res, s_apis in zip(queries, clarifier_res, similar_apis):
            results.append({
                "title": q,
                "clarify_input": c_res.demo_input,
                "clarify_output": c_res.demo_output,
                "clarify_statement": c_res.statement,
                "similar_apis": ", ".join(f"{api.fullname}" for api in s_apis),
                "clarify_tokens": c_res.tokens,
            })
        await save_to_csv(results,
                          PathConfig.DATA_DIR / 'clarifier' / f'{dataset.name}_clarify.csv')
    return clarifier_res, similar_apis


async def batch_code(clarifier_res, similar_apis, queries, answers, dataset):
    coder_res = await CoderConfig.code(clarifier_res, similar_apis)
    # Save answer
    results = []
    for i, c_res in enumerate(coder_res):
        results.append({
            "title": queries[i],
            "answer": ", ".join(f"{api.fullname}" for api in answers[i]),
            "apis": ", ".join(f"{api.fullname}" for api in c_res.apis),
            "code": c_res.code,
            "add_info": c_res.add_info,
            "tokens": c_res.tokens,
        })
    await save_to_csv(results, PathConfig.DATA_DIR / 'answer' / f"{dataset.name}_coder_results.csv")
    return coder_res


async def main():
    dataset = dt.Dataset(dt.DatasetName.BIKER, 'test', 'filtered')
    queries = dataset.titles
    answers = dataset.answers

    clarifier_res, similar_apis = await batch_clarify(queries, dataset)

    coder_res = await batch_code(clarifier_res, similar_apis, queries, answers, dataset)

    # Calculate metrics
    seq_lists = [res.apis for res in coder_res]
    for i, seq in enumerate(seq_lists):
        seq_lists[i] = [api.fullname for api in seq]
    ans_lists = []
    for _, row in dataset.values.iterrows():
        ans_lists.append([api.fullname for api in row['answer']])
    calculator = Calculator(seq_lists, ans_lists)
    metrics = calculator.calculate_metrics_for_multiple_k([1, 3, 5])
    results = [{
        "NAME": dataset.name,
        "MRR": metrics.mrr,
        "MAP": metrics.map,
        "SuccessRate@1": metrics.successrate_at_ks[0],
        "SuccessRate@3": metrics.successrate_at_ks[1],
        "SuccessRate@5": metrics.successrate_at_ks[2],
    }]
    pd.DataFrame(results).to_csv(
        PathConfig.DATA_DIR / 'result' / "result.csv",
        index=False, encoding="utf-8"
    )


if __name__ == "__main__":
    asyncio.run(main())
