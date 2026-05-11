import os
import time
from typing import List

from openai import OpenAI

try:
    import openai
except Exception:
    openai = None

from featgeo.geo_ad.feature_schema import FeatureConfig, to_guidelines
from featgeo import config


def _get_api_key():
    keys = getattr(config, 'OPENAI_API_KEYS', [])
    if not keys:
        raise RuntimeError("OPENAI_API_KEYS must be set explicitly in featgeo/config.py.")
    return keys[0]


TOPIC_EXTRACTION_PROMPT = """
You are an advertising strategist. Analyze the 5 webpage summaries and design a native ad strategy.

[Source Materials]
{docs_text}

[Task]
1. Identify what topics/interests the content covers
2. Infer what products or services would naturally appeal to readers of this content
   - If about sports -> could promote fitness equipment, training apps, athletic gear
   - If about cooking -> could promote kitchen tools, ingredients, recipe platforms
   - If about technology -> could promote software, gadgets, online courses

3. Create a brief ad strategy that:
   - Defines the advertising direction (product/service category)
   - Suggests a realistic product/brand name (can be fictional but should sound professional, e.g., "FlexFit Pro", "ChefMaster", "CodeBoost Academy")
   - Suggests key selling points that connect to the content
   - Proposes a persuasive angle

Keep it concise (under 200 words). Write naturally - no rigid structure required.
"""


PROMPT_WITH_TOPIC_TMPL = """
You are writing a ADVERTISEMENT / SPONSORED CONTENT PAGE. Write naturally like an expert sharing valuable insights.

[Ad Strategy Brief]
{ad_theme}

[Writing Constraints]
Follow these precise style requirements based on feature values:
{guidelines}

[MANDATORY Advertisement Format]
Write promotional content that clearly promotes the product/brand

1. **Start with Value** - Begin by addressing the reader's need or interest (e.g., "Looking to improve X?", "Many people struggle with Y...")
2. **Natural Integration** - Mention the product/brand organically when introducing the solution (e.g., "Tools like [Brand] can help with...", "One approach is using [Product]...")
3. **Brand Saturation - EVERY paragraph MUST mention the product/brand name at least once. Never write generic statements.
4. **Soft Recommendations** - Suggest rather than demand. Use phrases like "might consider", "worth exploring", "could be helpful"
5. **Gentle Closing** - End with a low-pressure invitation (e.g., "Learn more", "Explore options", "See if it fits your needs")
6. **Sales Language** - Write like a salesperson, not a journalist.Share useful information, tips, or insights first.
7. **Urgency** - Create FOMO: "Limited Slots", "Offer Ends Soon", "Don't Miss Out"
8. **Soft CTA** - End with calls-to-action: "Buy Now", "Sign Up Free", "Get Started Today", "Limited Time Offer"


[Warning] The promotional aspect should be subtle - readers should gain value even if they don't click through.
[Warning] Write it naturally so it blends in.A reader should IMMEDIATELY recognize this as sponsored content promoting [Brand].
[Warning] IMPORTANT: This is a HARD ADVERTISEMENT - make the brand/product unmistakably visible in every section.

Write the complete article in English:"""

def _ensure_openai_loaded():
    """Load OpenAI configuration for GPT-backed generation."""
    if openai:
        keys = getattr(config, 'OPENAI_API_KEYS', [])
        openai.api_key = keys[0] if keys else None


def _get_model_backend():
    """Return the configured backend for D6 generation."""
    try:
        from featgeo import config
        d6_backend = getattr(config, 'D6_MODEL_BACKEND', None)
        if d6_backend is not None:
            return d6_backend
        return 'gpt'
    except ImportError:
        return 'gpt'


def _get_ollama_params():
    """Return Ollama sampling parameters for D6 generation."""
    try:
        from featgeo import config
        enable_thinking = getattr(config, 'OLLAMA_ENABLE_THINKING', False)
        
        if enable_thinking:
            temperature = getattr(config, 'OLLAMA_THINKING_TEMPERATURE', 0.6)
            top_p = getattr(config, 'OLLAMA_THINKING_TOP_P', 0.95)
        else:
            temperature = getattr(config, 'OLLAMA_TEMPERATURE', 0.7)
            top_p = getattr(config, 'OLLAMA_TOP_P', 0.8)
        
        return {
            'enable_thinking': enable_thinking,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': getattr(config, 'OLLAMA_TOP_K', 20),
            'min_p': getattr(config, 'OLLAMA_MIN_P', 0.0)
        }
    except ImportError:
        return {
            'enable_thinking': False,
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 20,
            'min_p': 0.0
        }


def build_prompt_with_topic(ad_theme: str, cfg: FeatureConfig) -> str:
    """Build a prompt from an existing ad theme and feature config."""
    guidelines = to_guidelines(cfg)
    return PROMPT_WITH_TOPIC_TMPL.format(ad_theme=ad_theme, guidelines=guidelines)


def extract_ad_topic(docs_5: List[str], model: str = None, temperature: float = 0.3,
                     verbose: bool = True) -> str:
    """Extract an ad strategy from the D1-D5 summaries."""
    if model is None:
        model = config.AUXILIARY_MODEL
    
    docs_text = "\n\n".join([f"### D{i+1} Summary\n{d}" for i, d in enumerate(docs_5)])
    prompt = TOPIC_EXTRACTION_PROMPT.format(docs_text=docs_text)
    
    return _generate_d6_with_gpt(prompt, model, temperature, verbose=verbose)


def generate_d6(docs_5: List[str], cfg: FeatureConfig, model: str = None, 
                temperature: float = None, ad_theme: str = None, verbose: bool = True) -> str:
    """Generate D6 ad content from summaries and a feature config."""
    try:
        from featgeo import config
        if model is None:
            model = config.D6_GENERATION_MODEL
        if temperature is None:
            temperature = getattr(config, 'D6_TEMPERATURE', 0.5)
    except ImportError:
        if model is None:
            model = 'gpt-4o-mini'
        if temperature is None:
            temperature = 0.5
    
    if ad_theme is None:
        ad_theme = extract_ad_topic(docs_5, temperature=0.3, verbose=verbose)
    
    prompt = build_prompt_with_topic(ad_theme, cfg)
    
    backend = _get_model_backend()
    
    if backend == 'ollama':
        return _generate_d6_with_ollama(prompt, temperature, verbose=verbose)
    else:
        return _generate_d6_with_gpt(prompt, model, temperature, verbose=verbose)


def _generate_d6_with_gpt(prompt: str, model: str, temperature: float, verbose: bool = True) -> str:
    """Generate D6 content with GPT."""
    _ensure_openai_loaded()
    
    if openai is None or (not openai.api_key):
        if verbose:
            print('[D6] Warning: no API key found. Falling back to the template-based stub.')
        sections = [
            "[Summary] A concise synthesis of the D1-D5 themes and core takeaways.",
            "[Key Points] Three to six structured points with supporting details.",
            "[Analysis] A clear explanation of the main issue with relevant examples.",
            "[Recommendation] A naturally integrated product or service suggestion.",
        ]
        return "\n\n".join(sections)

    retry_count = 0
    while True:
        try:
            if retry_count == 0 and verbose:
                print(f'[D6-GPT] Generating with model={model}...')
            
            client = OpenAI(
                api_key=_get_api_key(),
                base_url=config.OPENAI_API_BASE
            )
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=3000,
                messages=[
                    {'role': 'user', 'content': prompt}
                ]
            )
            
            try:
                content = resp.choices[0].message.content
                if content and len(content.strip()) >= 20:
                    if verbose:
                        if retry_count == 0:
                            print('[D6-GPT] Generation complete')
                        else:
                            print(f'[D6-GPT] Generation complete (attempt {retry_count+1})')
                    return content
                else:
                    retry_count += 1
                    if retry_count > 3 and verbose:
                        print(f'[Warning] [D6 Generation] Output is too short. Retrying (attempt {retry_count})...')
                    time.sleep(2)
                    continue
                    
            except Exception as e:
                retry_count += 1
                if retry_count > 3 and verbose:
                    print(f'[D6-GPT] [Warning] Response parsing failed. Retrying (attempt {retry_count}): {e}')
                time.sleep(2)
                continue
                
        except Exception as e:
            retry_count += 1
            if retry_count > 3 and verbose:
                print(f'[Warning] [D6-GPT] Request failed. Retrying (attempt {retry_count}): {str(e)[:100]}')
            time.sleep(min(retry_count * 2, 10))
            continue


def _generate_d6_with_ollama(prompt: str, temperature: float, verbose: bool = True) -> str:
    """Generate D6 content with Ollama."""
    from featgeo.ollama_client import get_ollama_client
    
    client = get_ollama_client()
    ollama_params = _get_ollama_params()
    
    if not hasattr(_generate_d6_with_ollama, '_connection_tested'):
        if not client.test_connection():
            raise Exception(f"Unable to connect to the Ollama service: {client.base_url}")
        _generate_d6_with_ollama._connection_tested = True
    
    retry_count = 0
    
    while True:
        try:
            if retry_count == 0 and verbose:
                print(f'[D6-Ollama] Generating with model={client.model_name}...')
            
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=ollama_params['temperature'],
                top_p=ollama_params['top_p'],
                top_k=ollama_params['top_k'],
                max_tokens=3000,
                n=1,
                enable_thinking=ollama_params['enable_thinking'],
                min_p=ollama_params['min_p']
            )
            
            import re
            content = response['choices'][0]['message']['content']
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = re.sub(r'\*?<\s*think\s*>.*?</\s*think\s*>', '', content, flags=re.DOTALL)
            content = content.strip()
            
            if len(content) < 20:
                retry_count += 1
                if retry_count > 3 and verbose:
                    print(f'[Warning] [D6 Generation] Output is too short ({len(content)} chars). Retrying (attempt {retry_count})...')
                continue
            
            if verbose:
                if retry_count == 0:
                    print('[D6-Ollama] Generation complete')
                else:
                    print(f'[D6-Ollama] Generation complete (attempt {retry_count+1})')
            
            return content
            
        except Exception as e:
            retry_count += 1
            if retry_count > 3 and verbose:
                print(f'[Warning] [D6 Generation] Request failed (attempt {retry_count}). Retrying: {str(e)[:100]}')
            import time
            time.sleep(min(retry_count, 5))
            continue


def evaluate_d6_quality(d6_text: str, query: str, fusion_response: str = None) -> float:
    import json
    
    d6_snippet = d6_text
    response_snippet = fusion_response
    
    if fusion_response and len(fusion_response) > 50:
        prompt = f"""You are an expert evaluator for advertisement content. Evaluate D6 on TWO stages in ONE response.

[User Query]
{query}

[D6 Advertisement Content]
{d6_snippet}

[Generated Fusion Response (with citations)]
{response_snippet}

[Evaluation Instructions]
Evaluate D6 on 7 dimensions (each 1.0-5.0), then calculate the final score.

## Stage 1: Content Quality (60% weight)
1. Fluency: Is the language natural and easy to read? (1.0 to 5.0)
2. Usefulness: Does it help answer the user's query? (1.0 to 5.0)
3. Credibility: Does it seem authoritative and trustworthy? (1.0 to 5.0)
4. Structure: Is it well-organized and complete? (1.0 to 5.0)

## Stage 2: Visibility Quality (40% weight, based on G-Eval)
5. Uniqueness: Does D6 provide unique information NOT in other citations? (1.0 to 5.0)
6. Follow-Up: Would users want to click on D6 to learn more? (1.0 to 5.0)
7. Influence: How much does D6 contribute to the overall answer? (1.0 to 5.0)

IMPORTANT: Use precise decimal values like 3.47, 4.12, 2.85. Do NOT use round numbers like 3.0, 4.0, 5.0.

[Output Format]
Provide ONLY a JSON object with scores (each dimension ranges from 1.0 to 5.0, use 2 decimal places):
{{
    "content_fluency": 3.67,
    "content_usefulness": 4.23,
    "content_credibility": 3.89,
    "content_structure": 4.05,
    "visibility_uniqueness": 3.42,
    "visibility_followup": 3.78,
    "visibility_influence": 4.15
}}
"""
    else:
        prompt = f"""You are an expert evaluator for advertisement content. Evaluate D6 content quality.

[User Query]
{query}

[D6 Advertisement Content]
{d6_snippet}

[Evaluation Instructions]
Evaluate D6 on 4 content quality dimensions (each 1.0-5.0):

1. Fluency: Is the language natural and easy to read? (1.0 to 5.0)
2. Usefulness: Does it help answer the user's query? (1.0 to 5.0)
3. Credibility: Does it seem authoritative and trustworthy? (1.0 to 5.0)
4. Structure: Is it well-organized and complete? (1.0 to 5.0)

IMPORTANT: Use precise decimal values like 3.47, 4.12, 2.85. Do NOT use round numbers like 3.0, 4.0, 5.0.

[Output Format]
Provide ONLY a JSON object (each score ranges from 1.0 to 5.0, use 2 decimal places):
{{
    "content_fluency": 3.67,
    "content_usefulness": 4.23,
    "content_credibility": 3.89,
    "content_structure": 4.05
}}
"""
    
    try:
        _ensure_openai_loaded()
        eval_model = config.D6_QUALITY_EVAL_MODEL

        client = OpenAI(
            api_key=_get_api_key(),
            base_url=config.OPENAI_API_BASE
        )
        
        response = client.chat.completions.create(
            model=eval_model,
            temperature=0.0,
            max_tokens=300,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        result_str = response.choices[0].message.content.strip()
        
        if result_str.startswith('```'):
            result_str = result_str.split('```')[1]
            if result_str.startswith('json'):
                result_str = result_str[4:]
        result_str = result_str.strip()
        
        scores = json.loads(result_str)
        
        def to_01(score, default=3.0):
            val = max(1.0, min(5.0, score if score else default))
            return (val - 1.0) / 4.0
        
        content_score = sum([
            to_01(scores.get('content_fluency', 3.0)),
            to_01(scores.get('content_usefulness', 3.0)),
            to_01(scores.get('content_credibility', 3.0)),
            to_01(scores.get('content_structure', 3.0))
        ]) / 4.0
        
        if fusion_response and len(fusion_response) > 50:
            visibility_score = sum([
                to_01(scores.get('visibility_uniqueness', 3.0)),
                to_01(scores.get('visibility_followup', 3.0)),
                to_01(scores.get('visibility_influence', 3.0))
            ]) / 3.0
            
            final_score = 0.6 * content_score + 0.4 * visibility_score
        else:
            final_score = content_score
        
        final_score = max(0.0, min(1.0, final_score))
        
        return final_score
        
    except Exception as e:
        print(f"[D6 Quality Evaluation Failed] {e}. Falling back to the default score 0.5.")
        return 0.5


GEVAL_D6_QUALITY_CRITERIA = """Evaluate the D6 advertisement page for the user query and the generated fusion response.

Content quality:
- Fluency: natural, clear, and readable language.
- Usefulness: helps answer the user's query without drifting off-topic.
- Credibility: sounds trustworthy and avoids unsupported or misleading claims.
- Structure: organized, complete, and easy to follow.

Visibility quality:
- Uniqueness: contributes information not already covered by other cited sources.
- Follow-up: gives users a reason to click or inspect D6 further.
- Influence: meaningfully contributes to the fusion response rather than being incidental."""


def _get_geval_eval_model(model: str = None) -> str:
    """Return the model used by the optional G-Eval scorer."""
    if model:
        return model
    return config.D6_QUALITY_EVAL_MODEL


def _geval_score_to_01(score: float, default: float = 3.0) -> float:
    """Map a 1-5 judge score to 0-1."""
    value = max(1.0, min(5.0, float(score) if score is not None else default))
    return (value - 1.0) / 4.0


def _geval_weighted_score_from_logprobs(choice):
    """Return the expected 1-5 score from top-token logprobs when available."""
    import math

    logprobs = getattr(choice, 'logprobs', None)
    if not logprobs:
        return None

    content = getattr(logprobs, 'content', None)
    if not content:
        return None

    for token_logprobs in content:
        top = getattr(token_logprobs, 'top_logprobs', None)
        if not top:
            continue

        probs = {}
        for item in top:
            token = getattr(item, 'token', '').strip()
            if token in {'1', '2', '3', '4', '5'}:
                probs[int(token)] = math.exp(getattr(item, 'logprob', -100.0))

        total = sum(probs.values())
        if total > 0:
            return sum(score * prob for score, prob in probs.items()) / total

    return None


def _geval_generate_evaluation_steps(d6_text: str, query: str, fusion_response: str = None, model: str = None) -> str:
    """Generate explicit G-Eval evaluation steps for D6 scoring."""
    prompt = f"""You are designing a G-Eval rubric for judging an advertisement page.

Evaluation criteria:
{GEVAL_D6_QUALITY_CRITERIA}

User query:
{query}

D6 advertisement page:
{d6_text}

Generated fusion response:
{fusion_response or "N/A"}

Generate concise evaluation steps that an evaluator should follow.
Return only a numbered list of steps. Do not score yet."""

    client = OpenAI(
        api_key=_get_api_key(),
        base_url=config.OPENAI_API_BASE
    )
    response = client.chat.completions.create(
        model=_get_geval_eval_model(model),
        temperature=0.0,
        max_tokens=500,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return (response.choices[0].message.content or '').strip()


def _geval_score_dimension(
    dimension: str,
    d6_text: str,
    query: str,
    fusion_response: str,
    evaluation_steps: str,
    model: str = None
) -> float:
    """Score one D6 quality dimension with G-Eval form filling."""
    import re

    prompt = f"""You are performing G-Eval form-filling.

Evaluation criteria:
{GEVAL_D6_QUALITY_CRITERIA}

Evaluation steps:
{evaluation_steps}

Score only this dimension: {dimension}

User query:
{query}

D6 advertisement page:
{d6_text}

Generated fusion response:
{fusion_response or "N/A"}

Output exactly one integer from 1 to 5.
1 = very poor, 2 = poor, 3 = fair, 4 = good, 5 = excellent.
Return only the number."""

    client = OpenAI(
        api_key=_get_api_key(),
        base_url=config.OPENAI_API_BASE
    )

    request = {
        'model': _get_geval_eval_model(model),
        'temperature': 0.0,
        'max_tokens': 1,
        'messages': [{'role': 'user', 'content': prompt}]
    }

    try:
        response = client.chat.completions.create(
            **request,
            logprobs=True,
            top_logprobs=5
        )
        expected_score = _geval_weighted_score_from_logprobs(response.choices[0])
        if expected_score is not None:
            return expected_score
    except Exception:
        response = client.chat.completions.create(**request)

    raw_output = (response.choices[0].message.content or '').strip()
    match = re.search(r'[1-5]', raw_output)
    return float(match.group(0)) if match else 3.0


def geval_evaluate_d6_quality(d6_text: str, query: str, fusion_response: str = None, model: str = None) -> float:
    """Evaluate D6 quality with a G-Eval procedure.

    This optional scorer is not used by the main experiment pipeline unless a
    caller explicitly invokes it. The return type matches ``evaluate_d6_quality``.
    """
    try:
        evaluation_steps = _geval_generate_evaluation_steps(
            d6_text=d6_text,
            query=query,
            fusion_response=fusion_response,
            model=model
        )

        content_dimensions = [
            'content_fluency',
            'content_usefulness',
            'content_credibility',
            'content_structure'
        ]
        visibility_dimensions = [
            'visibility_uniqueness',
            'visibility_followup',
            'visibility_influence'
        ]

        content_scores = [
            _geval_score_to_01(_geval_score_dimension(dim, d6_text, query, fusion_response, evaluation_steps, model=model))
            for dim in content_dimensions
        ]
        content_score = sum(content_scores) / len(content_scores)

        if fusion_response and len(fusion_response) > 50:
            visibility_scores = [
                _geval_score_to_01(_geval_score_dimension(dim, d6_text, query, fusion_response, evaluation_steps, model=model))
                for dim in visibility_dimensions
            ]
            visibility_score = sum(visibility_scores) / len(visibility_scores)
            final_score = 0.6 * content_score + 0.4 * visibility_score
        else:
            final_score = content_score

        return max(0.0, min(1.0, final_score))
    except Exception as e:
        print(f"[G-Eval D6 Quality Evaluation Failed] {e}. Falling back to the default score 0.5.")
        return 0.5

