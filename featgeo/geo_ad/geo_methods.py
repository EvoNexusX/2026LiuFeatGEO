from featgeo.geo_functions import (
    fluent_optimization_gpt,
    unique_words_optimization_gpt,
    authoritative_optimization_mine,
    more_quotes_mine,
    citing_credible_sources_mine,
    simple_language_mine,
    technical_terms_mine,
    stats_optimization_mine,
    seo_optimize_mine2,
)


def identity(summary: str) -> str:
    """Return the source text unchanged."""
    return summary


GEO_METHODS = {
    "identity": identity,
    "fluent_gpt": fluent_optimization_gpt,
    "unique_words_gpt": unique_words_optimization_gpt,
    "authoritative_mine": authoritative_optimization_mine,
    "more_quotes_mine": more_quotes_mine,
    "citing_credible_mine": citing_credible_sources_mine,
    "simple_language_mine": simple_language_mine,
    "technical_terms_mine": technical_terms_mine,
    "stats_optimization_gpt": stats_optimization_mine,
    "seo_optimize_mine2": seo_optimize_mine2,
}


def extract_sources_from_dataset(dataset_sources, use_gpt_summary: bool = False):
    """Extract and pad source URLs and summaries from a GEO-Bench sample."""
    from featgeo.source_cleaner import clean_source_text_with_llm

    sources = []
    summaries = []

    for idx, source in enumerate(dataset_sources):
        sources.append(source.get("url", ""))

        if use_gpt_summary:
            print(f"  Cleaning source {idx + 1}/{len(dataset_sources)} with GPT...")
            raw_text = source.get("raw_text", "")[:8000]
            if raw_text:
                try:
                    summary = clean_source_text_with_llm(raw_text)
                except Exception as exc:
                    print(f"    GPT cleaning failed. Falling back to cleaned_text: {exc}")
                    summary = source.get("cleaned_text", "")[:8000]
            else:
                summary = source.get("cleaned_text", "")[:8000]
        else:
            summary = source.get("cleaned_text", "")
            if not summary:
                summary = source.get("raw_text", "")[:8000]

        summaries.append(summary)

    while len(summaries) < 5:
        summaries.append("")
        sources.append("")

    return sources, summaries

