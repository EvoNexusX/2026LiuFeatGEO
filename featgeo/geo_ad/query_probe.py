from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from featgeo.utils import extract_citations_new, get_answer
from featgeo import config
from openai import OpenAI


def _get_api_key():
    keys = getattr(config, 'OPENAI_API_KEYS', [])
    if not keys:
        raise RuntimeError("OPENAI_API_KEYS must be set explicitly in featgeo/config.py.")
    return keys[0]


def _load_config_value(name: str, default):
    """Read a config value while keeping this module importable in isolation."""
    try:
        from featgeo import config

        return getattr(config, name, default)
    except Exception:
        return default


def _get_openai_client():
    """Create an OpenAI-compatible client for probe-query generation."""
    return OpenAI(
        api_key=_get_api_key(),
        base_url=config.OPENAI_API_BASE
    )


def _extract_json_list(text: str) -> List[str]:
    """Extract a JSON string list from a model response."""
    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, list):
        raise ValueError("Probe-query response must be a JSON list")

    return [str(item).strip() for item in parsed if str(item).strip()]


def generate_probe_queries(
    query: str,
    total_count: int,
    *,
    include_original: bool = True,
    model: Optional[str] = None,
) -> List[str]:
    """Generate a topic-level set of semantically related probe queries."""
    total_count = max(1, int(total_count))
    generated_count = total_count - 1 if include_original else total_count

    probes: List[str] = [query] if include_original else []
    if generated_count <= 0:
        return probes[:total_count]

    if model is None:
        model = config.QUERY_PROBE_MODEL

    prompt = f"""Generate {generated_count} semantically related search queries for the same topic as the original query.

Original query:
{query}

Requirements:
- Preserve the same topic and user intent family.
- Vary wording, specificity, and perspective naturally.
- Do not optimize for any webpage or source.
- Return only a valid JSON array of strings, with no extra text."""

    response = _get_openai_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500,
        top_p=1,
    )

    generated = _extract_json_list(response.choices[0].message.content or "")

    seen = {item.lower() for item in probes}
    for item in generated:
        if item.lower() not in seen:
            probes.append(item)
            seen.add(item.lower())
        if len(probes) >= total_count:
            break

    return probes[:total_count]


def _extract_cited_source_ids(response_text: str, source_count: int) -> List[int]:
    """Return unique cited source ids in one fused answer."""
    cited = set()

    try:
        parsed = extract_citations_new(response_text)
        for paragraph in parsed:
            for sentence in paragraph:
                for source_id in sentence[2]:
                    if 1 <= source_id <= source_count:
                        cited.add(source_id)
    except Exception:
        for match in re.findall(r"\[(\d+)\]", response_text):
            source_id = int(match)
            if 1 <= source_id <= source_count:
                cited.add(source_id)

    return sorted(cited)


def rank_sources_by_probe_citations(
    probe_queries: Sequence[str],
    summaries: Sequence[str],
    *,
    num_completions: int = 1,
    print_info: bool = True,
) -> Tuple[List[int], Dict]:
    """Rank source indices by citation frequency across probe queries."""
    source_count = len(summaries)
    citation_counts = Counter({source_id: 0 for source_id in range(1, source_count + 1)})
    raw_answer_counts = Counter({source_id: 0 for source_id in range(1, source_count + 1)})
    per_query = []

    for probe_idx, probe_query in enumerate(probe_queries, 1):
        if print_info:
            print(f"  [Query Probe] Fusion {probe_idx}/{len(probe_queries)}: {probe_query}")

        try:
            answer_record = get_answer(
                probe_query,
                summaries=list(summaries),
                n=source_count,
                num_completions=num_completions,
            )
            responses = answer_record.get("responses", [[]])[-1]
        except Exception as exc:
            if print_info:
                print(f"  [Query Probe Warning] Fusion failed for probe {probe_idx}: {exc}")
            responses = []

        query_cited = set()
        for response_text in responses:
            cited_ids = _extract_cited_source_ids(response_text, source_count)
            query_cited.update(cited_ids)
            raw_answer_counts.update(cited_ids)

        citation_counts.update(query_cited)
        per_query.append(
            {
                "query": probe_query,
                "cited_source_ids": sorted(query_cited),
                "response_count": len(responses),
            }
        )

    ranked_source_ids = sorted(
        range(1, source_count + 1),
        key=lambda source_id: (-citation_counts[source_id], source_id),
    )

    info = {
        "probe_queries": list(probe_queries),
        "citation_counts": {str(k): int(v) for k, v in citation_counts.items()},
        "raw_answer_counts": {str(k): int(v) for k, v in raw_answer_counts.items()},
        "per_query": per_query,
        "ranked_source_ids": ranked_source_ids,
    }
    return ranked_source_ids, info


def select_probe_feature_summaries(
    query: str,
    summaries: Sequence[str],
    *,
    top_n: int,
    probe_count: int,
    include_original: bool = True,
    num_completions: int = 1,
    model: Optional[str] = None,
    print_info: bool = True,
) -> Tuple[List[str], Dict]:
    """Select the top-N cited source summaries for initial feature extraction."""
    source_count = len(summaries)
    if source_count == 0:
        return [], {
            "enabled": True,
            "probe_queries": [],
            "selected_source_ids": [],
            "citation_counts": {},
            "cache_tag": "query_probe:v1:empty",
        }

    top_n = max(1, min(int(top_n), source_count))

    try:
        probe_queries = generate_probe_queries(
            query,
            probe_count,
            include_original=include_original,
            model=model,
        )
    except Exception as exc:
        if print_info:
            print(f"  [Query Probe Warning] Probe-query generation failed: {exc}")
            print("  [Query Probe] Falling back to the original query only.")
        probe_queries = [query]

    ranked_source_ids, info = rank_sources_by_probe_citations(
        probe_queries,
        summaries,
        num_completions=num_completions,
        print_info=print_info,
    )

    selected_source_ids = ranked_source_ids[:top_n]
    selected_summaries = [summaries[source_id - 1] for source_id in selected_source_ids]
    cache_tag = build_query_probe_cache_tag(
        selected_source_ids,
        probe_count=len(probe_queries),
        top_n=top_n,
        num_completions=num_completions,
        include_original=include_original,
    )

    info.update(
        {
            "enabled": True,
            "top_n": top_n,
            "probe_count": len(probe_queries),
            "include_original": include_original,
            "num_completions": num_completions,
            "selected_source_ids": selected_source_ids,
            "cache_tag": cache_tag,
        }
    )

    if print_info:
        counts = info["citation_counts"]
        count_text = ", ".join(f"D{source_id}={counts[str(source_id)]}" for source_id in range(1, source_count + 1))
        selected_text = ", ".join(f"D{source_id}" for source_id in selected_source_ids)
        print(f"  [Query Probe] Citation frequency: {count_text}")
        print(f"  [Query Probe] Selected feature exemplars: {selected_text}")

    return selected_summaries, info


def build_query_probe_cache_tag(
    selected_source_ids: Sequence[int],
    *,
    probe_count: int,
    top_n: int,
    num_completions: int,
    include_original: bool,
) -> str:
    """Build a stable cache tag for a query-probe feature source selection."""
    selected = "-".join(str(source_id) for source_id in selected_source_ids)
    original_flag = "with_original" if include_original else "without_original"
    return (
        f"query_probe:v1:{original_flag}:"
        f"probes{probe_count}:top{top_n}:ncomp{num_completions}:sources{selected}"
    )

