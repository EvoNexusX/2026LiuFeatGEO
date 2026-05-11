import pickle
import time
import uuid

import nltk

from openai import OpenAI
from featgeo import config


_client = None

def _get_api_key():
    keys = getattr(config, 'OPENAI_API_KEYS', [])
    if not keys:
        raise RuntimeError("OPENAI_API_KEYS must be set explicitly in featgeo/config.py.")
    return keys[0]


def _get_client():
    global _client
    if _client is None:
        api_key = _get_api_key()
        base_url = config.OPENAI_API_BASE
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def clean_source_text_with_llm(source: str) -> str:
    response = None
    for idx in range(8):
        try:
            response = _get_client().chat.completions.create(
                model=getattr(config, "SOURCE_CLEANING_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "user",
                        "content": f"Clean and refine the extracted text from a website. Remove any unwanted content such as headers, sidebars, and navigation menus. Retain only the main content of the page and ensure that the text is well-formatted and free of HTML tags, special characters, and any other irrelevant information. Refined text should contain the main intended readable text. Apply markdown formatting when outputting the answer.\n\nHere is the website:\n```html_text\n{source.strip()}```",
                    },
                ],
                temperature=0,
                max_tokens=1800,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            break
        except Exception as exc:
            print(f"Error while cleaning text with openai {exc}")
            source = source[:-int(800 * (1 + idx / 2))]
            time.sleep(3 + idx**2)

    if response is None:
        raise RuntimeError("Failed to clean source text after repeated retries.")

    from pathlib import Path
    Path("response_usages").mkdir(exist_ok=True)
    try:
        pickle.dump(response.usage, open(f"response_usages/{uuid.uuid4()}.pkl", "wb"))
    except Exception:
        pass

    try:
        text = response.choices[0].message.content.strip()
    except (AttributeError, KeyError):
        try:
            text = response.choices[0]["message"]["content"].strip()
        except Exception:
            text = str(response.choices[0]).strip()

    merged_lines = [""]
    for line in text.split("\n\n"):
        merged_lines[-1] += line + "\n"
        if len(nltk.sent_tokenize(line)) != 1:
            merged_lines.append("")
    merged_lines = [line.strip() for line in merged_lines]
    return "\n\n".join(merged_lines)

