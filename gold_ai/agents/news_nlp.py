"""News NLP pipeline (Section 5.Q).

Deliberately NOT a naive positive/negative headline classifier. Pipeline:

1. normalize text, extract topics/entities (lexicon-based, deterministic)
2. deduplicate stories reporting the same event (shingle Jaccard clustering)
3. classify: BREAKING / FOLLOW_UP / OLD_INFO / COMMENTARY / OPINION /
   CONFIRMED_EVENT / RUMOR_UNVERIFIED
4. score per-article: credibility(source tier), freshness, novelty, relevance
   to gold/USD/rates, sentiment toward GOLD, expected macro impact, urgency,
   potential market impact
5. unverified info gets very low reliability and can never drive a strong view
"""
from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timedelta

from gold_ai.core.enums import NewsClassification, SourceTier
from gold_ai.core.models import NewsArticle
from gold_ai.core.timeutils import age_seconds

# ---------------------------------------------------------------------------
GOLD_TERMS = ["gold", "xau", "bullion", "goldetf", "glld", "preclous metal", "precious metal",
              "aurum", "gold reserve", "miner"]
USD_TERMS = ["dollar", "usd", "dxy", "greenback", "buck", "foreign exchange", "fx"]
RATES_TERMS = ["yield", "treasury", "treasuries", "bond", "rate", "fomc", "fed funds", "tips",
               "real yield", "curve"]
FED_TERMS = ["fed", "fomc", "powell", "central bank", "federal reserve", "rate decision"]
GEO_TERMS = ["war", "strike", "missile", "sanction", "conflict", "invasion", "ceasefire",
             "escalat", "military", "nuclear", "tariff", "geopolit", "unrest", "coup"]
OPINION_MARKERS = ["opinion", "view", "commentary", "editorial", "column", "should", "believe",
                   "think", "argue"]
RUMOR_MARKERS = ["unverified", "reportedly", "rumor", "sources say", "alleged", "unconfirmed",
                 "said to be", "whispers"]
BULLISH_FOR_GOLD = ["safe haven", "haven demand", "reserve purchase", "buying", "inflows",
                    "escalation", "uncertainty", "recession risk", "dovish", "weaker dollar",
                    "dollar slips", "yields retreat", "real yields fall", "inflation rises",
                    "cooling labor", "claims rise", "debt ceiling", "banking stress", "crisis"]
BEARISH_FOR_GOLD = ["risk-on", "rally in stocks", "equity surge", "hawkish", "stronger dollar",
                    "dollar climbs", "yields climb", "yields rise", "real yields rise",
                    "disinflation", "outflows", "selling", "profit taking", "record high yields",
                    "firm payrolls", "resilient economy"]


def _shingles(text: str, k: int = 4) -> set[str]:
    toks = re.findall(r"[a-z]+", text.lower())
    return {" ".join(toks[i:i + k]) for i in range(max(1, len(toks) - k + 1))}


def similarity(a: NewsArticle, b: NewsArticle) -> float:
    sa, sb = getattr(a, "_sh", None) or _shingles(a.headline), getattr(b, "_sh", None) or _shingles(b.headline)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class NewsPipeline:
    """Stateful stream processor: dedup memory + classification of new items."""

    def __init__(self, dedup_window_s: float = 6 * 3600, sim_threshold: float = 0.35) -> None:
        self.dedup_window_s = dedup_window_s
        self.sim_threshold = sim_threshold
        self.seen: list[NewsArticle] = []

    # ------------------------------------------------------------------
    def process(self, art: NewsArticle, now: datetime) -> NewsArticle:
        hl = art.headline.lower()
        body = (art.body or "").lower()
        text = f"{hl} {body}"

        # topics & entities
        topics = list(art.topics)
        for name, terms in (("gold", GOLD_TERMS), ("usd", USD_TERMS), ("rates", RATES_TERMS),
                            ("fed", FED_TERMS), ("geopolitics", GEO_TERMS)):
            if any(t in text for t in terms) and name not in topics:
                topics.append(name)
        art.topics = topics
        art.entities = sorted({w for w in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", art.headline)})[:8]

        # relevance scores
        art.relevance_gold = min(1.0, sum(t in text for t in GOLD_TERMS) * 0.35 + 0.15 * ("gold" in topics))
        art.relevance_usd = min(1.0, sum(t in text for t in USD_TERMS) * 0.3)
        art.relevance_rates = min(1.0, sum(t in text for t in RATES_TERMS) * 0.25)

        # sentiment toward GOLD (context lexicon, not generic polarity)
        bull = sum(1 for t in BULLISH_FOR_GOLD if t in text)
        bear = sum(1 for t in BEARISH_FOR_GOLD if t in text)
        raw = (bull - bear) / max(1, bull + bear)
        art.sentiment = float(raw)

        # rumor / opinion detection
        is_rumor = any(m in text for m in RUMOR_MARKERS) or not art.verified
        is_opinion = any(m in text for m in OPINION_MARKERS) or art.tier == SourceTier.TIER_3 and "commentary" in " ".join(topics)

        # deduplication vs recent stories
        dup_of: NewsArticle | None = None
        best_sim = 0.0
        cutoff = now - timedelta(seconds=self.dedup_window_s)
        self.seen = [s for s in self.seen if s.effective_at >= cutoff]
        for s in self.seen:
            sim = similarity(s, art)
            if sim > best_sim:
                best_sim, dup_of = sim, s
        art.dedup_key = hashlib.sha1(re.sub(r"\W+", "", hl)[:120].encode()).hexdigest()[:16]

        age = age_seconds(art.effective_at, now)
        freshness = math.exp(-age / 3600.0)

        if dup_of is not None and best_sim >= self.sim_threshold:
            if (now - dup_of.effective_at).total_seconds() < 90:
                art.classification = NewsClassification.FOLLOW_UP
            else:
                art.classification = NewsClassification.OLD_INFORMATION
            art.novelty = max(0.03, 0.25 * (1 - best_sim))
        elif is_rumor:
            art.classification = NewsClassification.RUMOR_UNVERIFIED
            art.novelty = 0.7
        elif is_opinion:
            art.classification = NewsClassification.OPINION
            art.novelty = 0.35
        elif art.potential_market_impact >= 0.7 and freshness > 0.6:
            art.classification = NewsClassification.BREAKING
            art.novelty = 0.95
        elif art.classification == NewsClassification.CONFIRMED_EVENT:
            art.novelty = 0.8
        else:
            art.classification = NewsClassification.MARKET_COMMENTARY
            art.novelty = 0.4

        # urgency & impact
        art.urgency = float(min(1.0, freshness * (0.5 + 0.5 * art.potential_market_impact)))
        cred = art.reliability_score()
        if art.classification == NewsClassification.RUMOR_UNVERIFIED:
            cred = min(cred, 0.25)   # unverified => very low reliability, hard cap
        art.nlp_reliability = cred * (0.3 + 0.7 * art.novelty)
        art.expected_macro_impact = float(min(1.0, art.potential_market_impact * (0.4 + 0.6 * art.relevance_gold)))
        art.potential_market_impact = float(min(1.0, art.expected_macro_impact * art.nlp_reliability + 0.05))

        self.seen.append(art)
        return art

    # ------------------------------------------------------------------
    def aggregate(self, now: datetime, window_min: int = 180) -> dict:
        """Aggregate current news signal for the news agent."""
        cutoff = now - timedelta(minutes=window_min)
        items = [a for a in self.seen if a.effective_at >= cutoff]
        if not items:
            return {"score": 0.0, "impact": 0.0, "breaking": False, "count": 0,
                    "top": [], "unverified_count": 0}
        # weighted by relevance*reliability*impact; Tier-3 capped so it can never
        # independently generate a strong view.
        num = 0.0
        den = 0.0
        top = []
        unverified = 0
        for a in items:
            w = a.relevance_gold * a.nlp_reliability * (0.3 + a.potential_market_impact)
            if a.classification == NewsClassification.RUMOR_UNVERIFIED:
                w *= 0.15
                unverified += 1
            num += a.sentiment * w
            den += w
            top.append((w, a))
        top.sort(key=lambda x: -x[0])
        score = float(num / den) if den > 1e-9 else 0.0
        impact = float(min(1.0, den / 1.2))
        breaking = any(a.classification == NewsClassification.BREAKING for _, a in top[:3])
        return {
            "score": score,
            "impact": impact,
            "breaking": breaking,
            "count": len(items),
            "unverified_count": unverified,
            "top": [{"headline": a.headline[:110], "sent": round(a.sentiment, 2),
                     "cls": a.classification.value, "rel": round(a.nlp_reliability, 2),
                     "imp": round(a.potential_market_impact, 2),
                     "tier": a.tier.value, "age_s": round(age_seconds(a.effective_at, now))}
                    for _, a in top[:6]],
        }
