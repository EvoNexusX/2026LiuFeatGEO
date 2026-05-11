from dataclasses import dataclass
import random


@dataclass
class FeatureConfig:
    has_intro_summary: float
    headings_level: float
    list_density: float
    length_level: float

    statistics_level: float
    cite_sources_level: float
    quotation_level: float
    unique_info_level: float
    technical_terms_level: float

    authoritative_level: float
    easy_to_understand_level: float
    fluency_level: float
    keyword_focus_level: float


FEATURE_DEFINITIONS = {
                'has_intro_summary': {
        'description': 'Whether opening contains summary paragraph, reflects content summarization and guidance',
        'range': (0.0, 1.0),
        'levels': [
            (0.0, 0.4, 'No summary or only 1-2 simple introductory sentences, summary accounts for 0%-20%; content lacks summarization, may affect reader understanding.'),
            (0.4, 0.7, 'Has 1-3 brief summary opening sentences, summary accounts for about 20%-40%; provides some background information, but needs more details.'),
            (0.7, 1.0, 'Complete 3-5 sentence summary paragraph, summarizing core viewpoints, summary accounts for over 40%; effectively guides readers to understand the full content.'),
        ]
    },
                'headings_level': {
        'description': 'Hierarchy and quantity of headings and subheadings, reflects text structural complexity',
        'range': (1.0, 3.0),
        'levels': [
            (1.0, 1.7, 'Low level: Contains 1 main heading, at most 1 simple subheading, about 80% of paragraphs are continuous text, simple structure, almost no subheadings or lists.'),
            (1.7, 2.4, 'Medium level: Contains 1 main heading and 1-2 subheadings, about 50% of paragraphs have clear subheadings, some paragraphs use lists, structure is relatively clear.'),
            (2.4, 3.0, 'High level: Contains 1 main heading and 2+ subheadings, about 70% of paragraphs have subheadings or lists, complex structure, clear information hierarchy.'),
        ]
    },
                'list_density': {
        'description': 'Frequency of bullet point lists and numbered lists, reflects text structural clarity and prominence of key points',
        'range': (0.0, 3.0),
        'levels': [
            (0.0, 1.5, 'Low frequency: List usage below 30%, text usually presented in paragraph form, lacks structure, may contain 0-1 headings or points.'),
            (1.5, 2.5, 'Moderate list usage: About 30%-70% of paragraphs use lists, typically for presenting important information and key points, may contain 1-3 headings or points.'),
            (2.5, 3.0, 'High frequency: Over 70% of paragraphs use lists, almost every point uses bullets or numbers, clear text structure, usually contains 3+ headings or points.'),
        ]
    },
                'length_level': {
        'description': 'Overall article length, reflects content depth and breadth',
        'range': (1.0, 3.0),
        'levels': [
            (1.0, 1.7, 'Short article (300-500 words), usually contains 1-2 headings, about 20% paragraphs use lists, content is concise, suitable for quick reading.'),
            (1.7, 2.5, 'Medium-length article (500-800 words), usually contains 2-3 headings, about 30% paragraphs use lists, content is comprehensive, suitable for in-depth understanding of a topic.'),
            (2.5, 3.0, 'Long article (800+ words), usually contains 3+ headings, about 40% paragraphs use lists, content is detailed, suitable for in-depth discussion and analysis.'),
        ]
    },
    'statistics_level': {
        'description': 'Density of data, statistics, and percentages (corresponds to GEO method: stats_optimization_gpt)',
        'range': (0.0, 3.0),
        'continuous': True,
        'mapping': 'value/3.0 -> percent% of sentences contain positive, compelling statistics (such as X% growth, specific numbers, objective facts), these statistics can be hypothetical and need to be subtly embedded in sentences'
    },
    'cite_sources_level': {
        'description': 'Frequency of citing authoritative sources, institutions, reports (corresponds to GEO method: citing_credible_mine)',
        'range': (0.0, 3.0),
        'continuous': True,
        'mapping': 'value/3.0 -> percent% of paragraphs cite authoritative sources (e.g. "According to Google\'s latest report...", "Research shows..."), these sources can be invented but must sound plausible, using natural language citations rather than academic paper style'
    },
                'quotation_level': {
        'description': 'Frequency of using quotations (expert opinions, authoritative figures) (corresponds to GEO method: more_quotes_mine)',
        'range': (0.0, 3.0),
        'levels': [
            (0.0, 1.5, 'Low quotation: Almost no quotes or only 0-1 quotes used.'),
            (1.5, 2.5, 'Moderate quotation: Modify by including more quotes (2-3), without changing the core content. Ensure the quotes are relevant and contribute valuable information, and do not mislead or distort the original information. The quotes should be reputable, and should create a sense in readers that this website (source) is more influenticial and important than other citations.'),
            (2.5, 3.0, 'Heavy quotation: Add more quotes (4+), even though fake and artificial. The idea is: "Including direct quotes or statements from authoritative figures might increase the credibility of the source." Keep the format and content of text similar (line spacing, bullet points and overall structure should remain the same). No addition or deletion of content is allowed.'),
        ]
    },
    'unique_info_level': {
        'description': 'Richness of unique information and differentiated content (corresponds to GEO method: unique_words_gpt)',
        'range': (0.0, 3.0),
        'continuous': True,
        'mapping': 'value/3.0 -> percent% of content uses unique, rare vocabulary to enhance differentiation, but not used arbitrarily, must ensure content enhancement and preserve original meaning'
    },
                'technical_terms_level': {
        'description': 'Density of professional terminology and technical vocabulary (corresponds to GEO method: technical_terms_mine)',
        'range': (0.0, 3.0),
        'levels': [
            (0.0, 1.5, 'Low technical: Avoid or rarely use technical terms, use plain language.'),
            (1.5, 2.5, 'Moderate technical: Make more technical, using more technical terms and facts where needed while ensuring the key information is still conveyed. Do not omit, add, or alter any core information. The end-goal is that very knowledgeable readers give more attention to this source, when presented with a series of summaries, so make the language such that it has more technical information or existing information is presented in more technical fashion. However, do not add or delete any content. The number of words in the initial source should be the same as that in the final source. Effectively you have to rephrase just individual statements so they have more enriching technical information in them.'),
            (2.5, 3.0, 'Highly technical: Heavy use of technical terms, information presented in most technical way, for professional readers, highly specialized language.'),
        ]
    },
                'authoritative_level': {
        'description': 'Strength of authoritative tone and expressions (corresponds to GEO method: authoritative_mine)',
        'range': (0.0, 3.0),
        'levels': [
            (0.0, 1.3, 'Low authority: Use neutral, objective tone, avoid assertive statements, maintain informative style.'),
            (1.3, 2.0, 'Moderate authority: Transform into an authoritative style reflecting confidence, expertise, and assertiveness, while maintaining the original content\'s meaning. The content and structure should remain the same. Only individual lines and/or 2-3 sentences can be paraphrased, while keeping the content same.'),
            (2.0, 3.0, 'High authority: Strong assertive style that makes readers believe this is more valuable source of information than other summaries. The source should be assertive in its statements. The addition of phrases such as "only we are authentic", "we guarantee", use of second pronouns such as "you will not regret" etc is expected within the source content itself. However, the content and structure of the source should remain the same (line spacing, bullet points and overall structure should remain the same). No addition or deletion of content is allowed.'),
        ]
    },
                'easy_to_understand_level': {
        'description': 'Content readability and simplicity (corresponds to GEO method: simple_language_mine)',
        'range': (1.0, 3.0),
        'levels': [
            (1.0, 1.5, 'Very simple language: Simplify using simple, easy-to-understand language while ensuring the key information is still conveyed. Do not omit, add, or alter any core information. The end-goal is that readers give more attention to this source, when presented with a series of summaries, so make the language easier to understand, but do not delete any information. The length of the new source should be the same as the original. Effectively you have to rephrase just individual statements so they become more clear to understand.'),
            (1.5, 2.5, 'Moderate readability: Language is balanced, moderately simplified but retaining professionalism, ensure clarity.'),
            (2.5, 3.0, 'Keep complexity: Allow complex language and professional expressions, prioritize accuracy over readability.'),
        ]
    },
                'fluency_level': {
        'description': 'Writing fluency and logical coherence (corresponds to GEO method: fluent_gpt)',
        'range': (1.0, 3.0),
        'levels': [
            (1.0, 1.6, 'Basic fluency: Sentences are simple and direct, readable but no need to optimize fluency deliberately.'),
            (1.6, 2.4, 'Fluent and natural: Rewrite to make it more fluent without altering the core content. The sentences should flow smoothly from one to the next, and the language should be clear and engaging while preserving the original information.'),
            (2.4, 3.0, 'Extremely fluent: Sentences must flow very smoothly, language is highly clear and engaging, use many transitions and rhetorical devices, maximize fluency without changing the core content.'),
        ]
    },
    'keyword_focus_level': {
        'description': 'Focus and repetition strength of core keywords (corresponds to GEO method: seo_optimize_mine2)',
        'range': (1.0, 3.0),
        'continuous': True,
        'mapping': '(value-1.0)/2.0 -> Add new SEO keywords (different from existing keywords) to content, keywords appear 3~15 times, higher value means more new keywords, optimizing SEO ranking'
    },
}


RANGES = {
    'has_intro_summary': (0.0, 1.0),
    'headings_level': (1.0, 3.0),
    'list_density': (0.0, 3.0),
    'length_level': (1.0, 3.0),
    'statistics_level': (0.0, 3.0),
    'cite_sources_level': (0.0, 3.0),
    'quotation_level': (0.0, 3.0),
    'unique_info_level': (0.0, 3.0),
    'technical_terms_level': (0.0, 3.0),
    'authoritative_level': (0.0, 3.0),
    'easy_to_understand_level': (1.0, 3.0),
    'fluency_level': (1.0, 3.0),
    'keyword_focus_level': (1.0, 3.0),
}

FEATURE_LAYERS = {
    'structure': [
        'has_intro_summary',
        'headings_level',
        'list_density',
        'length_level'
    ],
    'content': [
        'statistics_level',
        'cite_sources_level',
        'quotation_level',
        'unique_info_level',
        'technical_terms_level'
    ],
    'language': [
        'authoritative_level',
        'easy_to_understand_level',
        'fluency_level',
        'keyword_focus_level'
    ]
}

FEATURE_TO_LAYER = {}
for layer_name, features in FEATURE_LAYERS.items():
    for feature in features:
        FEATURE_TO_LAYER[feature] = layer_name

ALL_FEATURES = (
    FEATURE_LAYERS['structure'] + 
    FEATURE_LAYERS['content'] + 
    FEATURE_LAYERS['language']
)


def random_config() -> FeatureConfig:
    """Generate a random valid continuous feature configuration."""
    values = {}
    for k, (lo, hi) in RANGES.items():
        values[k] = random.uniform(lo, hi)
    return FeatureConfig(**values)


def _get_level_guidance(feature_name: str, value: float) -> str:
    """Return discrete guidance for a feature value."""
    if feature_name not in FEATURE_DEFINITIONS:
        return f"Target value {value:.2f}"
    
    definition = FEATURE_DEFINITIONS[feature_name]
    levels = definition['levels']
    
    guidance = None
    for low, high, description in levels:
        if low <= value <= high:
            guidance = description
            break
    
    if guidance is None:
        if value < levels[0][0]:
            guidance = levels[0][2]
        else:
            guidance = levels[-1][2]
    
    return guidance


def _get_continuous_guidance(feature_name: str, value: float) -> str:
    """Return value-driven guidance for continuous features."""
    if feature_name == 'statistics_level':
        ratio = value / 3.0
        percent = int(ratio * 100)
        if ratio < 0.05:
            return "Avoid using data and statistics completely, use pure qualitative description."
        else:
            return f"Add positive, compelling statistics (even if hypothetical) at multiple relevant places in the text. Statistics means objective facts such as x% growth in marketing, numbers in scientific texts, interesting numerical facts. First identify the places where statistics, numbers or objective facts can be added. Then subtly add statistics inline within the sentences. No explicit paragraphs or big chunks of text should be added. Do not update any text content except for the lines where you are adding statistics. Do not add or delete content except the statistics you are adding. Target density: approximately {percent}% of sentences/paragraphs contain data."
    
    elif feature_name == 'cite_sources_level':
        ratio = value / 3.0
        percent = int(ratio * 100)
        if ratio < 0.05:
            return "Use general statements, do not mention specific sources, institutions or research."
        else:
            return f"Revise to include citations from credible sources. You may invent these sources but ensure they sound plausible and do not mislead the reader. Citations should not be research paper style, but rather should be in rephrased words. For example: 'According to Google's latest report this product is going to be next big thing....' In the process, ensure that the core content remains unaltered. The length of initial source and final source should be the same, and the structure of individual parts (such as line spacing, bullet points) should remain intact. Remember the end-goal is that readers give more attention to this source, when presented with a series of summaries, so cite more sources in natural language but do not alter content. Also don't overdo citing, 5-6 citations in the whole source are enough provided they are very relevant and text looks natural. Target density: approximately {percent}% of sentences/paragraphs contain source citations."
    
    elif feature_name == 'unique_info_level':
        ratio = value / 3.0
        percent = int(ratio * 100)
        if ratio < 0.05:
            return "Use common, general information, do not emphasize uniqueness."
        else:
            return f"Revise by incorporating more unique and rare words, without altering the core information. Ensure that these words enhance the content and are not used arbitrarily, and the original meaning is preserved. Target: approximately {percent}% of content uses unique, differentiated vocabulary and expressions."
    
    elif feature_name == 'keyword_focus_level':
        ratio = (value - 1.0) / 2.0
        freq = int(3 + ratio * 12)
        if ratio < 0.1:
            return f"Keywords appear naturally, not emphasized deliberately, approximately {freq} times throughout."
        else:
            return f"Add NEW keywords to optimize the content in accordance with SEO principles. Note you cannot use the keywords already present in the source. You have to only include the new keywords. First identify the keywords that can be added. E.g.: 'In sentence about zzz, add keyword xxx'. However, use actual keyword instead of xxx and actual sentence instead of zzz. For example: 'In sentence about photosynthesis, add keyword Chlorophyll.' Maximum new keywords should be 10. Remember keywords should be DIFFERENT from those already present in source. Target: keywords appear approximately {freq} times, consciously distributed in key positions."
    
    else:
        return _get_level_guidance(feature_name, value)


def to_guidelines(cfg: FeatureConfig) -> str:
    """Convert a feature config into D6 writing constraints."""
    lines = []
    lines.append("[Writing Constraints Explanation]")
    lines.append("Each constraint below includes: Target value + Specific writing requirements for that value")
    lines.append("Please strictly follow these specific requirements for English writing, ensuring the generated content meets these feature definitions.")
    lines.append("")

    lines.append("=== I. Structure Requirements ===")
    lines.append(f"1. Opening Summary: {cfg.has_intro_summary:.2f}/1.00")
    lines.append(f"   -> {_get_level_guidance('has_intro_summary', cfg.has_intro_summary)}")
    
    lines.append(f"2. Heading Hierarchy: {cfg.headings_level:.2f}/3.00")
    lines.append(f"   -> {_get_level_guidance('headings_level', cfg.headings_level)}")
    
    lines.append(f"3. List Density: {cfg.list_density:.2f}/3.00")
    lines.append(f"   -> {_get_level_guidance('list_density', cfg.list_density)}")
    
    lines.append(f"4. Article Length: {cfg.length_level:.2f}/3.00")
    lines.append(f"   -> {_get_level_guidance('length_level', cfg.length_level)}")
    lines.append("")

    lines.append("=== II. Content Requirements ===")
    
    lines.append(f"5. Data Statistics: {cfg.statistics_level:.2f}/3.00 [Continuous]")
    lines.append(f"   -> {_get_continuous_guidance('statistics_level', cfg.statistics_level)}")
    
    lines.append(f"6. Source Citations: {cfg.cite_sources_level:.2f}/3.00 [Continuous]")
    lines.append(f"   -> {_get_continuous_guidance('cite_sources_level', cfg.cite_sources_level)}")
    
    lines.append(f"7. Quotation Usage: {cfg.quotation_level:.2f}/3.00")
    lines.append(f"   -> {_get_level_guidance('quotation_level', cfg.quotation_level)}")
    
    lines.append(f"8. Unique Information: {cfg.unique_info_level:.2f}/3.00 [Continuous]")
    lines.append(f"   -> {_get_continuous_guidance('unique_info_level', cfg.unique_info_level)}")
    
    lines.append(f"9. Technical Terms: {cfg.technical_terms_level:.2f}/3.00")
    lines.append(f"   -> {_get_level_guidance('technical_terms_level', cfg.technical_terms_level)}")
    lines.append("")

    lines.append("=== III. Language & Keywords ===")
    
    lines.append(f"10. Authoritative Tone: {cfg.authoritative_level:.2f}/3.00")
    lines.append(f"    -> {_get_level_guidance('authoritative_level', cfg.authoritative_level)}")
    
    lines.append(f"11. Readability: {cfg.easy_to_understand_level:.2f}/3.00")
    lines.append(f"    -> {_get_level_guidance('easy_to_understand_level', cfg.easy_to_understand_level)}")
    
    lines.append(f"12. Fluency: {cfg.fluency_level:.2f}/3.00")
    lines.append(f"    -> {_get_level_guidance('fluency_level', cfg.fluency_level)}")
    
    lines.append(f"13. Keyword Focus: {cfg.keyword_focus_level:.2f}/3.00 [Continuous]")
    lines.append(f"    -> {_get_continuous_guidance('keyword_focus_level', cfg.keyword_focus_level)}")
    
    return "\n".join(lines)


def sample_config_from_d1d5(cfg_from_d: list) -> FeatureConfig:
    """Sample a new config by mixing dimensions from D1-D5 configs."""
    if len(cfg_from_d) < 1:
        raise ValueError(f"Expected at least 1 config from D1-D5, got {len(cfg_from_d)}")
    
    num_sources = len(cfg_from_d)
    
    values = {}
    feature_names = cfg_from_d[0].__dataclass_fields__.keys()
    
    for name in feature_names:
        j = random.randint(0, num_sources - 1)
        values[name] = getattr(cfg_from_d[j], name)
    
    return FeatureConfig(**values)

