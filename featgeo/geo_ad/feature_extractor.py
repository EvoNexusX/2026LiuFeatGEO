import json
import re
from openai import OpenAI
from featgeo.geo_ad.feature_schema import FeatureConfig, FEATURE_DEFINITIONS
from featgeo import config


def _get_api_key():
    keys = getattr(config, 'OPENAI_API_KEYS', [])
    if not keys:
        raise RuntimeError("OPENAI_API_KEYS must be set explicitly in featgeo/config.py.")
    return keys[0]


def _build_feature_evaluation_guide() -> str:
    """Build a GPT-readable scoring guide from ``FEATURE_DEFINITIONS``."""
    guide_lines = []
    
    feature_order = [
        'has_intro_summary', 'headings_level', 'list_density', 'length_level',
        'statistics_level', 'cite_sources_level', 'quotation_level', 
        'unique_info_level', 'technical_terms_level',
        'authoritative_level', 'easy_to_understand_level', 
        'fluency_level', 'keyword_focus_level'
    ]
    
    for idx, feat_name in enumerate(feature_order, 1):
        if feat_name not in FEATURE_DEFINITIONS:
            continue
            
        definition = FEATURE_DEFINITIONS[feat_name]
        desc = definition['description']
        range_val = definition['range']
        
        guide_lines.append(f"\n{idx}. **{feat_name}** ({range_val[0]}-{range_val[1]})")
        guide_lines.append(f"   Definition: {desc}")
        
        if 'levels' in definition:
            guide_lines.append("   Scoring ranges:")
            for low, high, level_desc in definition['levels']:
                guide_lines.append(f"   - [{low:.1f}-{high:.1f}]: {level_desc}")
        
        elif 'continuous' in definition and definition['continuous']:
            mapping = definition.get('mapping', '')
            guide_lines.append(f"   Continuous mapping: {mapping}")
    
    return '\n'.join(guide_lines)


def infer_config_from_text(text: str) -> FeatureConfig:
    """Infer a ``FeatureConfig`` from a text sample with GPT."""
    feature_guide = _build_feature_evaluation_guide()
    
    analysis_prompt = f"""
Analyze the text below using the standardized feature definitions and provide
an objective numeric evaluation.

Text to analyze:
```
{text}
```

Feature scoring guide:
{feature_guide}

Output requirements:
Return only valid JSON in the following format, with no extra text:

```json
{{
  "has_intro_summary": <0.0-1.0>,
  "headings_level": <1.0-3.0>,
  "list_density": <0.0-3.0>,
  "length_level": <1.0-3.0>,
  "statistics_level": <0.0-3.0>,
  "cite_sources_level": <0.0-3.0>,
  "quotation_level": <0.0-3.0>,
  "unique_info_level": <0.0-3.0>,
  "technical_terms_level": <0.0-3.0>,
  "authoritative_level": <0.0-3.0>,
  "easy_to_understand_level": <1.0-3.0>,
  "fluency_level": <1.0-3.0>,
  "keyword_focus_level": <1.0-3.0>
}}
```

Important:
- Follow the standardized definitions exactly when assigning scores.
- Choose values based on the actual observed text characteristics.
- Decimal values such as 0.5, 1.8, and 2.3 are allowed.
- Return JSON only, with no surrounding explanation.
    """

    try:
        client = OpenAI(
            api_key=_get_api_key(),
            base_url=config.OPENAI_API_BASE
        )
        feature_model = config.FEATURE_EXTRACTION_MODEL
        
        response = client.chat.completions.create(
            model=feature_model,
            temperature=0.3,
            max_tokens=500,
            messages=[
                {'role': 'user', 'content': analysis_prompt}
            ]
        )
        
        gpt_output = response.choices[0].message.content.strip()
        
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', gpt_output, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(gpt_output)
        
        return FeatureConfig(
            has_intro_summary=float(result.get('has_intro_summary', 0.5)),
            headings_level=float(result.get('headings_level', 2.0)),
            list_density=float(result.get('list_density', 1.5)),
            length_level=float(result.get('length_level', 2.0)),
            statistics_level=float(result.get('statistics_level', 1.5)),
            cite_sources_level=float(result.get('cite_sources_level', 1.5)),
            quotation_level=float(result.get('quotation_level', 1.0)),
            unique_info_level=float(result.get('unique_info_level', 2.0)),
            technical_terms_level=float(result.get('technical_terms_level', 1.5)),
            authoritative_level=float(result.get('authoritative_level', 2.0)),
            easy_to_understand_level=float(result.get('easy_to_understand_level', 2.5)),
            fluency_level=float(result.get('fluency_level', 2.5)),
            keyword_focus_level=float(result.get('keyword_focus_level', 2.0))
        )
        
    except Exception as e:
        print(f"  [Warning] GPT feature analysis failed. Falling back to rule-based defaults: {e}")
        return _infer_config_from_text_fallback(text)


def _infer_config_from_text_fallback(text: str) -> FeatureConfig:
    """Fallback heuristic when GPT-based inference fails."""
    lines = text.split('\n')
    char_count = len(text)
    
    first_para = lines[0] if lines else ""
    has_intro = 0.8 if len(first_para) > 100 else 0.3
    
    heading_count = len([l for l in lines if l.strip().startswith('#')])
    headings = 1.0 if heading_count == 0 else (2.0 if heading_count <= 3 else 3.0)
    
    list_items = len(re.findall(r'^\s*[-*\d]+[\.\)]\s', text, re.MULTILINE))
    list_density = 0.5 if list_items == 0 else (1.5 if list_items <= 5 else 2.5)
    
    length = 1.0 if char_count < 500 else (2.0 if char_count < 1500 else 3.0)
    
    number_matches = len(re.findall(r'\d+%|\d+\.\d+|\d{4}', text))
    statistics = min(3.0, number_matches / 3.0)
    
    citation_patterns = len(re.findall(r'according to|research|study|report', text, re.IGNORECASE))
    cite_sources = min(3.0, citation_patterns / 2.0)
    
    quote_count = text.count('"') + text.count('"')
    quotation = 0.5 if quote_count == 0 else (2.0 if quote_count <= 4 else 3.0)
    
    return FeatureConfig(
        has_intro_summary=has_intro,
        headings_level=headings,
        list_density=list_density,
        length_level=length,
        statistics_level=statistics,
        cite_sources_level=cite_sources,
        quotation_level=quotation,
        unique_info_level=2.0,
        technical_terms_level=2.0,
        authoritative_level=2.0,
        easy_to_understand_level=2.5,
        fluency_level=2.5,
        keyword_focus_level=2.0
    )

