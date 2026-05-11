import json
import math
import itertools
import os
from featgeo import config

PROMPT_TEMPLATE = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_msg}[/INST]"

def get_prompt(source, query):
    system_prompt = """You are a helpful, respectful and honest assistant.
Given a web source, and context, your only purpose is to summarize the source, and extract topics that may be relevant to the context. Even if a line is distinctly relevant to the context, include that in the summary. It is preferable to pick chunks of text, instead of isolated lines.
"""

    user_msg = f"### Context: ```\n{query}\n```\n\n ### Source: ```\n{source}\n```\n Now summarize the text in more than 1000 words, keeping in mind the context and the purpose of the summary. Be as detailed as possible.\n"

    return PROMPT_TEMPLATE.format(system_prompt=system_prompt, user_msg=user_msg)

import re
import nltk



def get_num_words(line):
    return len([x for x in line if len(x)>2])

def extract_citations_new(text):
    """Extract citation markers from generated response text."""
    def ecn(sentence):
        """Extract cited source indices from one sentence."""
        citation_pattern = r'\[[^\w\s]*\d+[^\w\s]*\]'
        return [int(re.findall(r'\d+', citation)[0]) for citation in re.findall(citation_pattern, sentence)]

    paras = re.split(r'\n\n', text)
    sentences = [nltk.sent_tokenize(p) for p in paras]
    words = [[(nltk.word_tokenize(s), s, ecn(s)) for s in sentence] for sentence in sentences]
    return words

def impression_wordpos_count_simple(sentences, n = 5, normalize=True):
    """Compute citation influence using word count and positional decay."""
    sentences = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    
    for i, sent in enumerate(sentences):
        for cit in sent[2]:  # sent[2] stores cited source ids such as [1, 2, 6].
            score = get_num_words(sent[0])
            score *= math.exp(-1 * i / (len(sentences)-1)) if len(sentences)>1 else 1
            score /= len(sent[2])  # Split credit across co-cited sources.

            try: scores[cit-1] += score  # Citation ids are 1-based.
            except: print(f'Citation Hallucinated: {cit}')
    
    return [x/sum(scores) for x in scores] if normalize and sum(scores)!=0 else [1/n for _ in range(n)] if normalize else scores

def impression_word_count_simple(sentences, n = 5, normalize=True):
    sentences = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    for i, sent in enumerate(sentences):
        for cit in sent[2]:
            score = get_num_words(sent[0])
            score /= len(sent[2])
            try: scores[cit-1] += score
            except: print(f'Citation Hallucinated: {cit}')
    return [x/sum(scores) for x in scores] if normalize and sum(scores)!=0 else [1/n for _ in range(n)] if normalize else scores
    

def impression_pos_count_simple(sentences, n = 5, normalize=True):
    sentences = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    for i, sent in enumerate(sentences):
        for cit in sent[2]:
            score = 1
            score *= math.exp(-1 * i / (len(sentences)-1)) if len(sentences)>1 else 1
            score /= len(sent[2])
            try: scores[cit-1] += score
            except: print(f'Citation Hallucinated: {cit}')
    return [x/sum(scores) for x in scores] if normalize and sum(scores)!=0 else [1/n for _ in range(n)] if normalize else scores
                

def impression_para_based(sentences, n = 5, normalize = True, alpha = 1.1, beta = 1.5, gamma2 = 1/math.e):
    scores = [0 for _ in range(n)]
    power_scores = [1 for _ in range(n)]
    average_lines = sum([len(x) for x in sentences])/len(sentences)
    for i, para in enumerate(sentences):
        citation_counts = [0 for _ in range(n)]
        for sent in para:
            for c in sent[2]:
                try:
                    citation_counts[c-1] += get_num_words(sent[0])
                except Exception:
                    print(f"Citation Hallucinated: {c}")
        if sum(citation_counts)==0:
            continue
        
        for cit_num, cit in enumerate(citation_counts):
            if cit==0: continue
            score = cit/sum(citation_counts)
            
            score *= beta**(len(para)/average_lines - 1)
            
            if i == 0:
                score *= 1
            elif i != len(sentences)-1:
                score *= math.exp(-1 * i / (len(sentences)-2))
            else:
                score *= gamma2

            try:
                power_scores[cit_num] *= (alpha) ** (cit/sum(citation_counts))
                scores[cit_num] += score
            except:
                print(f'Citation Hallucinated: {cit}')
         
    final_scores = [x*y for x, y in zip(scores, power_scores)]
    return [x/sum(final_scores) for x in final_scores] if normalize and sum(final_scores)!=0 else final_scores


CACHE_FILE = config.GLOBAL_CACHE_FILE


def set_cache_file(path: str):
    """Set the fusion cache file for the current process."""
    global CACHE_FILE
    CACHE_FILE = path

from featgeo.generative_le import generate_answer

def check_summaries_exist(sources, summaries):
    for source in sources:
        s2 = [x['summary'] for x in source['sources']]  
        if s2 == summaries:
            return source
    return None

def get_answer(query, summaries = None, n = 5, num_completions = 1, cache_idx = 0, regenerate_answer = False, write_to_cache = True, loaded_cache = None):
    """Generate or reuse fusion responses for one query and summary set."""
    if loaded_cache is None:
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'r') as f:
                    cache = json.load(f)
            else:
                cache = {}
        except (json.JSONDecodeError, IOError) as e:
            print(f"[Warning] Cache file is corrupted or unreadable ({CACHE_FILE}): {e}. Using an empty cache.")
            cache = {}
    else:
        cache = loaded_cache
    if summaries is None:
        raise ValueError("summaries must be provided; web scraping mode has been removed.")
    
    if cache.get(query) is None:
        cache[query] = []
    
    cached_source = check_summaries_exist(cache[query], summaries)
    if not regenerate_answer and cached_source is not None:
        if len(cached_source['responses']) > 0:
            print('[Cache Hit] Reusing cached GPT fusion responses')
            answers = cached_source['responses'][-1]
            return cached_source
        else:
            answers = generate_answer(query, summaries, num_completions = num_completions) 
    else:
        answers = generate_answer(query, summaries, num_completions = num_completions) 
    ret_value = None
    if loaded_cache is None:
        try:
            # Reload the latest on-disk cache before writing.
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'r') as f:
                    cache = json.load(f)
            else:
                cache = {}
        except (json.JSONDecodeError, IOError) as e:
            print(f"[Warning] Cache file is corrupted or unreadable ({CACHE_FILE}): {e}. Using an empty cache.")
            cache = {}
    else:
        cache = loaded_cache

    if cache.get(query) is None:
        cache[query] = [{'sources': [{'summary' : x} for x in summaries], 'responses': [answers]}]
    else:
        flag = False
        for source in cache[query]:
            s2 = [x['summary'] for x in source['sources']]  
            if s2 == summaries:
                source['responses'].append(answers)
                ret_value = source
                flag = True
                break
        if not flag:
            cache[query].append({'sources': [{'summary' : x, 'source' : y} for x, y in zip(summaries, cache[query][0]['sources'])], 'responses': [answers]})
    if write_to_cache:
        try:
            # Atomic write: write to a temp file and then replace the target.
            temp_file = CACHE_FILE + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(cache, f, indent=2)
            os.replace(temp_file, CACHE_FILE)
        except Exception as e:
            print(f"[Warning] Failed to write the cache file ({CACHE_FILE}): {e}")

    return ret_value if ret_value is not None else cache[query][-1] 

