import json
import os
import pickle
import uuid
import re
import time

from openai import OpenAI
from featgeo import config

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

COMMON_SYSTEM_PROMPT = """You are an expert ml researcher having previous background in SEO and search engines in general. You are working on novel research ideas for next generation of products. These products will have language models augmented with search engines, with the task of answering questions based on sources backed by the search engine. This new set of systems will be collectively called language engines (generative search engines). This will require websites to update their SEO techniques to rank higher in the llm generated answer. Specifically they will use GEO (Generative Engine Optimization) techniques to boost their visibility in the final text answer outputted by the Language Engine.  
"""


global_cache = None
GEO_CACHE_FILE = config.GEO_CACHE_FILE


def set_geo_cache_file(path: str):
    """Set the GEO rewrite cache file for the current process."""
    global GEO_CACHE_FILE, global_cache
    GEO_CACHE_FILE = path
    global_cache = None


def call_gpt(user_prompt, system_prompt = COMMON_SYSTEM_PROMPT, model = None, temperature = 0.0, num_completions = 1, regenerte_answer = False, pre_msgs = None):
    """Run a GEO rewrite request with caching."""
    global global_cache
    
    if model is None:
        try:
            from featgeo import config
            model = config.AUXILIARY_MODEL
        except ImportError:
            model = 'gpt-4o-mini'
    
    CACHE_FILE = GEO_CACHE_FILE
    
    cache_model_name = model
    cache_file = CACHE_FILE.replace('.json',f'_{cache_model_name}.json')
    
    if not os.path.exists(cache_file):
        with open(cache_file, 'w') as f:
            json.dump({}, f)
    
    if config.STATIC_CACHE:
        if global_cache is None:
            try:
                global_cache = json.load(open(cache_file, 'r'))
            except (json.JSONDecodeError, ValueError):
                print(f"[Warning] Cache file is corrupted. Recreating: {cache_file}")
                global_cache = {}
                with open(cache_file, 'w') as f:
                    json.dump({}, f)
        cache = global_cache
    else:
        try:
            cache = json.load(open(cache_file, 'r'))
        except (json.JSONDecodeError, ValueError):
            print(f"[Warning] Cache file is corrupted. Recreating: {cache_file}")
            cache = {}
            with open(cache_file, 'w') as f:
                json.dump({}, f)
    if str((user_prompt, system_prompt)) in cache and not regenerte_answer:
        print('[Cache Hit] Reusing cached GEO optimization output')
        return cache[str((user_prompt, system_prompt))][-1]

    print('[Cache Miss] Running GEO optimization with GPT...')

    messages = [
        { 'role': "system", 'content': system_prompt },
        { 'role': "user", 'content': user_prompt }
    ]

    if pre_msgs is not None:
        messages = [messages[0]] + pre_msgs + messages[1:]

    for attempt in range(10):
        try:
            responses = get_openai_client().chat.completions.create(
                model = model,
                temperature=temperature,
                max_tokens=4096,
                messages = messages,
                top_p=1,
                n=num_completions,
            )
            break
        except Exception as e:
            print('Error',e)
            print(str(e).find('maximum context length'))
            if str(e).find('maximum context length')!=-1:
                try:
                    a = str(e).find('messages resulted in ') + len('messages resulted in ')
                    b = str(e).find(' tokens', a)
                    num_tokens_excess = 2000 / int(str(e)[a:b])
                except:
                    a = str(e).find('you requested ') + len('you requested ')
                    b = str(e).find(' tokens', a)
                    num_tokens_excess = 2000 / int(str(e)[a:b])
                    
                print(num_tokens_excess)
                lup = len(messages[-1]['content'])
                messages[-1]['content'] = messages[-1]['content'][:int(lup * num_tokens_excess)]  
            if attempt > 5:
                messages[0]['content'][:-200]       
                messages[-1]['content'][:-1000]       
            print(messages)
            import time
            time.sleep(15)
            continue
    
    os.makedirs("response_usages_16k", exist_ok=True)
    pickle.dump(responses.usage, open(f"response_usages_16k/{uuid.uuid4()}.pkl", "wb"))

    if global_cache is None:
        # Re-read the cache defensively in case another process updated it.
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        except (json.JSONDecodeError, IOError, FileNotFoundError) as e:
            print(f"[Warning] Failed to read cache file ({cache_file}): {e}. Using an empty cache.")
            cache = {}
    if str((user_prompt, system_prompt)) not in cache:
        cache[str((user_prompt, system_prompt))] = []
    def get_summary(tex):
        tex = tex.replace('```\n```','```')
        b = tex.rfind('```')
        if b!=-1:
            if tex.count('```')<2:
                a = b + 3
                b = -1
            else:
                a = tex[:b].rfind('```') + 3
        else:
            a = -1
        if b-a < 50:
            a = b if len(tex) - b > 200 else a
            b = -1

        if a <= 2: a = 0
        if b!=-1:
            new_tex = tex[a:b].strip()
        else:
            new_tex = tex[a:].strip()
        if new_tex.lower().startswith('updated'):
            new_tex = '\n'.join(new_tex.splitlines()[1:])
        if len(new_tex) == 0:
            return tex
        return new_tex
    
    try:
        if not hasattr(responses, 'choices') or not responses.choices:
            print('[Warning] GPT response is invalid: missing choices.')
            print(f'Response object type: {type(responses)}')
            raise ValueError("Invalid GPT response structure")
        
        contents = []
        for idx, choice in enumerate(responses.choices):
            try:
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content
                elif isinstance(choice, dict) and 'message' in choice:
                    content = choice['message']['content']
                else:
                    content = str(choice)
                
                contents.append(get_summary(content))
            except Exception as e:
                print(f'[Warning] Failed to process choice #{idx}: {e}')
                raise
                
    except Exception as e:
        print(f'[Error] Failed to process GPT response: {e}')
        print(f'Response type: {type(responses)}')
        raise
    
    cache[str((user_prompt, system_prompt))].extend(contents)
    
    # Write the cache atomically to avoid partial writes.
    try:
        temp_file = cache_file + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(cache, f, indent=2)
        os.replace(temp_file, cache_file)
    except Exception as e:
        print(f"[Warning] Failed to write cache ({cache_file}): {e}")
    if config.STATIC_CACHE:
        global_cache = cache
    return cache[str((user_prompt, system_prompt))][-1]




def fluent_optimization_gpt(summary):
    user_prompt = f"""User Prompt: "Rewrite the following source to make it more fluent without altering the core content. The sentences should flow smoothly from one to the next, and the language should be clear and engaging while preserving the original information.

Source: {summary}"
"""
    return call_gpt(user_prompt)

def unique_words_optimization_gpt(summary):
    user_prompt = f"""User Prompt: "Revise the following source by incorporating more unique and rare words, without altering the core information. Ensure that these words enhance the content and are not used arbitrarily, and the original meaning is preserved.

Source: {summary}"
"""
    return call_gpt(user_prompt)

def authoritative_optimization_mine(summary):
    user_prompt = """Transform the following source into an authoritative style without adding or removing any core information. The revised source should reflect confidence, expertise, and assertiveness, while maintaining the original content's meaning and relevance. 
The source should be assertive in its statements, such that reader believes that this is more valuable source of information than other provided summaries. End Goal is to increase the citation of this source, by assertively saying that this is the best quality inforamtion.
However, the content and structure of the source should remain the same. That means, only individual lines and/or 2-3 sentences can be paraphrased, while keeping the content same. 

Source: 
```
{summary}
```

Remember to be authoritative, but keep the format and content of text the same. 
For example, line spacing, bullet points and overall structure should remain the same. 
No addition or deletion of content is allowed. 
Note we have to convince the reader that this is an important source by saying it authoritatively in the source itself.
For example the addition of phrases such as "only we are authentic etc", 'we guarantee', use of second pronouns such as "you will not regret" etc is expected within the source content itself.""".format(summary = summary).strip()
    return call_gpt(user_prompt)

def more_quotes_mine(summary):
    user_prompt = """Modify the following source by including more quotes, without changing the core content. Ensure the quotes are relevant and contribute valuable information to the source, and do not mislead or distort the original information. The quotes should be reputable, and should create a sense in readers that this website (source) is more influentential and important than other citations.

Source: 
```
{summary}
```

Add more quotes in the source, even though fake and artificial. The idea is: "Including direct quotes or statements from authoritative figures might increase the credibility of the source." 
Keep the format and content of text similar. 
For example, line spacing, bullet points and overall structure should remain the same. 
No addition or deletion of content is allowed. """.format(summary = summary).strip()
    
    return call_gpt(user_prompt)

def citing_credible_sources_mine(summary):
    user_prompt = """Revise the following source to include citations from credible sources. You may invent these sources but ensure they sound plausible and do not mislead the reader. Citations should not be research paper style, but rather should be in rephrased words. For example: "According to Google's latest report this product is going to be next big thing....' 
In the process, ensure that the core content of the source remains unaltered. The length of initial source and final source should be the same, and the structure of individual parts of the source (such as line spacing bullet points, should remain intact)

Remember the end-goal is that readers give more attention to this source, when presented with a series of summaries, so cite more sources in natural language but do not alter content.

Source: 
```
{summary}
```

Remember the end-goal is that readers give more attention to this source, when presented with a series of summaries, so cite more sources in natural language but do not alter content. Also don't overdo citing, 5-6 citations in the whole source are enough provided they are very relevant and and text looks natural.""".format(summary = summary).strip()
    
    return call_gpt(user_prompt)

def simple_language_mine(summary):
    user_prompt = """Simplify the following source, using simple, easy-to-understand language while ensuring the key information is still conveyed. Do not omit, add, or alter any core information in the process. 

Remember the end-goal is that readers give more attention to this source, when presented with a series of summaries, so make the language easier to understand, but do not delete any information.
The length of the new source should be the same as the original. Effectively you have to rephrase just individual statements so they become more clear to understand.

Source: 
```
{summary}
```
""".format(summary = summary).strip()
    
    return call_gpt(user_prompt)

def technical_terms_mine(summary):
    user_prompt = """Make the following source more technical, using giving more technical terms and facts where needed while ensuring the key information is still conveyed. Do not omit, add, or alter any core information in the process. 

Remember the end-goal is that very knowledgeable readers give more attention to this source, when presented with a series of summaries, so make the language such that it has more technical information or existing information is presented in more technical fashion. However, Do not add or delete any content . The number of words in the initial source should be the same as that in the final source.
The length of the new source should be the same as the original. Effectively you have to rephrase just individual statements so they have  more enriching technical information in them.

Source:
{summary}
""".format(summary = summary).strip()
    
    return call_gpt(user_prompt)

def seo_optimize_mine2(summary):

    user_prompt = """Here is the source that you need to update:
```
{summary}
```

## Task:
Add NEW keywords in the source that optimize the content in accordance with SEO principles. Note you cannot use the keywords already present in the source. You have to only include the new keywords.

## Guidelines to follow:
1. Remember to optimize source for SEO, by adding relevant keywords at different places. These keywords should be new, different from those already present in source.
2. First identify the keywords that can be added. Eg: "In sentence about zzz, add keyword xxx". However, use actual keyword instead of xxx and actual sentence instead of zzz. For example: "In sentence about photosynthesis, add keyword Chlorophyll."
3. Maximum new keywords should be 10. Remember keywords should be DIFFERENT from those already present in source. 
4. Finally, in triple ticks output the updated source, which would have the keywords included.

## Output Format: 
1. In sentence about keyword zzz, add keyword xxx
2. In sentence about keyword zzz, add keyword xxx
....
k. In sentence about keyword zzz, add keyword xxx

Now I will output the updated text:
Updated Output:
```
<Output>
```""".format(summary = summary).strip()

    return call_gpt(user_prompt)

def stats_optimization_mine(summary):


    user_prompt = """Here is the source:
```
{summary}
```

## Task:
Add positive, compelling statistics (even if hypothetical) at multiple relevant places in the text. Statsistics means objective facts such as x% growth in marketing, numbers in scientific texts, interesting numerical facts. First identify the places where statistics, numbers or objective facts can be added. Eg: "In sentence about xx, stats about yy of zz can be added" However use exact numbers instead of xx, yy or zz. Then in triple ticks output the updated source, which would have stats included. 

## Guidelines to follow:
1. Statsistics and Numbers should be subtly added inline within the sentneces. No explicit paragraphs or big chunks of text should be added.
2. Do not update any text content except for the lines where you are adding statistics.
3. Do not add or delete content except the statistics you are adding. Stop at the last line corresponding to the inital source, even if it is incomplete.
4. Just output the optimized source text. No need to give any explanation or reasoning or conclusion.
5. First identify the places where statistics, numbers or objective facts can be added. Eg: "In sentence about xx, stats about yy of zz can be added". However use exact numbers instead of xx, yy or zz. Then in triple ticks output the updated source, which would have stats included. 


## Output Format: 
1. Stat to be added
2. Stat to be added.
....
k. Stat to be added.

Updated Output:
```
<Output>
```
""".format(summary = summary).strip()
    return call_gpt(user_prompt)

