import uuid
from pathlib import Path

from openai import OpenAI
from featgeo import config

query_prompt = """Write an accurate and concise answer for the given user question, using _only_ the provided summarized web search results. The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user's language of choice such as English, Français, Español, Deutsch, or 日本語 should be used. The answer should be informative, interesting, and engaging. The answer's logic and reasoning should be rigorous and defensible. Every sentence in the answer should be _immediately followed_ by an in-line citation to the search result(s). The cited search result(s) should fully support _all_ the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You can use multiple search results to respond comprehensively while avoiding irrelevant search results.

Question: {query}

Search Results:
{source_text}
"""


client = None


def _get_api_key():
    keys = getattr(config, 'OPENAI_API_KEYS', [])
    if not keys:
        raise RuntimeError("OPENAI_API_KEYS must be set explicitly in featgeo/config.py.")
    return keys[0]


def get_openai_client():
    global client
    if client is None:
        api_key = _get_api_key()
        base_url = config.OPENAI_API_BASE
        client = OpenAI(api_key=api_key, base_url=base_url)
    return client


def _validate_fusion_response(content: str, n_sources: int) -> bool:
    """Return whether a fused response passes basic citation checks."""
    if len(content) < 20:
        return False

    has_any_citation = False
    for i in range(1, n_sources + 1):
        if f'[{i}]' in content:
            has_any_citation = True
            break

    if not has_any_citation:
        return False

    for i in range(n_sources + 1, 20):
        if f'[{i}]' in content:
            return False

    if 'cannot fulfill' in content.lower() or 'unable to' in content.lower():
        return False

    return True


def generate_answer(query, sources, num_completions, temperature=0.5, verbose=False, model=None):
    """Generate cited answers from multiple source summaries."""
    if model is None:
        try:
            from featgeo import config
            model = config.FUSION_MODEL
        except ImportError:
            model = 'gpt-4o-mini'

    source_text = '\n\n'.join(['### Source ' + str(idx + 1) + ':\n' + source + '\n\n\n' for idx, source in enumerate(sources)])
    prompt = query_prompt.format(query=query, source_text=source_text)

    return _generate_with_gpt(prompt, num_completions, temperature, model, verbose, len(sources))


def _generate_with_gpt(prompt, num_completions, temperature, model, verbose, n_sources=6):
    """Generate fused answers with GPT and bounded retries."""
    all_valid_results = []
    retry_count = 0
    max_retries = 20

    while len(all_valid_results) < num_completions and retry_count < max_retries:
        try:
            remaining = num_completions - len(all_valid_results)
            batch_size = max(remaining, int(remaining * 1.5))

            if retry_count == 0:
                if verbose:
                    print('[GPT] Calling the OpenAI API...')

            response = get_openai_client().chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=2048,
                messages=[
                    {'role': "user", 'content': prompt}
                ],
                top_p=1,
                n=batch_size
            )

            if verbose and retry_count == 0:
                print('[GPT] Generation complete')

            if retry_count == 0:
                Path("response_usages_16k").mkdir(exist_ok=True)
                try:
                    import pickle
                    pickle.dump(response.usage, open(f"response_usages_16k/{uuid.uuid4()}.pkl", "wb"))
                except Exception:
                    pass

            try:
                results = [x.message.content + '\n' for x in response.choices]
            except (AttributeError, KeyError):
                try:
                    results = [x['message']['content'] + '\n' for x in response.choices]
                except Exception:
                    results = [x.get('text', x.get('message', {}).get('content', '')) + '\n' for x in response.choices]

            batch_results = []
            for content in results:
                if _validate_fusion_response(content, n_sources):
                    batch_results.append(content)
                elif retry_count > 3 and verbose:
                    print('[Warning] [GPT] Invalid fusion response filtered out. Retrying...')

            all_valid_results.extend(batch_results)

            if len(all_valid_results) >= num_completions:
                if verbose:
                    print('[GPT] Generation complete')
                return all_valid_results[:num_completions]

            retry_count += 1
            if retry_count > 3 and retry_count % 5 == 0:
                print(f'[Warning] [GPT] Attempt {retry_count}: still collecting valid cited responses...')

        except Exception as e:
            print(f'[GPT] Error in calling OpenAI API: {e}')
            import time
            time.sleep(15)
            retry_count += 1
            continue

    if len(all_valid_results) < num_completions:
        print(f'[Warning] [GPT] Reached the retry limit ({max_retries}). Returning {len(all_valid_results)}/{num_completions} valid responses.')
    return all_valid_results
