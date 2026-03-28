#!/usr/bin/env python3
"""
text_predictor.py

This module converts *news text* into:
- aqi_txt          : pollution tone score (-50 to +50)
- confidence_txt   : certainty of the tone (0 to 1)
- tags_txt         : extracted pollution keywords

No ML training required.
Uses simple, stable heuristics based on embeddings and keywords.
"""

import numpy as np

# -------------------------
# pollution-related keywords
# -------------------------

POLLUTION_KEYWORDS = {
    "fire":        +35,
    "wildfire":    +40,
    "forest fire": +40,
    "smog":        +30,
    "haze":        +20,
    "dust":        +15,
    "dust storm":  +30,
    "pollution":   +25,
    "chemical":    +20,
    "leak":        +25,
    "gas leak":    +35,
    "factory":     +10,
    "industrial":  +10,
}

CLEAN_AIR_KEYWORDS = {
    "rain":       -20,
    "windy":      -10,
    "clear":      -15,
    "clean air":  -20,
    "improving":  -10,
    "air quality improving": -20,
}


# ------------------------------------------------------------
# Step 1: Keyword Extraction (tags_txt + partial aqi contribution)
# ------------------------------------------------------------

def extract_tags_and_keyword_score(headlines: str):
    text = headlines.lower() if headlines else ""
    tags = []
    score = 0.0

    for k, v in POLLUTION_KEYWORDS.items():
        if k in text:
            tags.append(k)
            score += v

    for k, v in CLEAN_AIR_KEYWORDS.items():
        if k in text:
            tags.append(k)
            score += v

    # clip to reasonable scale
    score = np.clip(score, -50, +50)

    return tags, score


# ------------------------------------------------------------
# Step 2: Embedding-Based Tone Estimation
# We compute “pollution polarity” using a hand-crafted pollution vector.
# ------------------------------------------------------------

# pollution direction vector (unit vector)
POLLUTION_VECTOR = np.array([
    0.61, 0.48, 0.22, 0.40, 0.25,
    0.55, 0.33, 0.17, 0.41, 0.29
])
POLLUTION_VECTOR = POLLUTION_VECTOR / np.linalg.norm(POLLUTION_VECTOR)


def embedding_tone(embedding):
    """
    embedding: list or np.array
    returns: a tone score (-30 to +30) and confidence (0 to 1)
    """

    if embedding is None or len(embedding) == 0:
        return 0.0, 0.0

    emb = np.array(embedding, dtype=float)

    # reduce or pad to size of POLLUTION_VECTOR
    if len(emb) < len(POLLUTION_VECTOR):
        emb = np.pad(emb, (0, len(POLLUTION_VECTOR)-len(emb)))
    else:
        emb = emb[:len(POLLUTION_VECTOR)]

    emb_norm = emb / (np.linalg.norm(emb) + 1e-8)

    # cosine similarity → pollution polarity
    cos_sim = float(np.dot(emb_norm, POLLUTION_VECTOR))

    # convert similarity → AQI tone
    tone = cos_sim * 30  # roughly -30 to +30

    # confidence = magnitude of similarity
    conf = min(abs(cos_sim), 1.0)

    return tone, conf


# ------------------------------------------------------------
# FINAL PUBLIC FUNCTION
# ------------------------------------------------------------

def predict_from_news(headlines: str, news_embedding):
    """
    Returns:
        aqi_txt        (float)
        confidence_txt (float)
        tags_txt       (list[str])
    """

    # 1. keyword pass
    tags, keyword_score = extract_tags_and_keyword_score(headlines)

    # 2. embedding pass
    emb_tone, emb_conf = embedding_tone(news_embedding)

    # 3. combine both
    aqi_txt = keyword_score + emb_tone
    aqi_txt = float(np.clip(aqi_txt, -50, +50))

    # confidence = embedding + keyword presence
    keyword_boost = 0.2 if len(tags) > 0 else 0.0
    confidence_txt = float(np.clip(emb_conf + keyword_boost, 0.0, 1.0))

    return aqi_txt, confidence_txt, tags


# ------------------------------------------------------------
# quick CLI test
# ------------------------------------------------------------
if __name__ == "__main__":
    sample_headlines = "Wildfire smoke causes severe smog across region."
    sample_embedding = [0.21, 0.35, 0.12, 0.40, 0.28, 0.44, 0.30, 0.11, 0.36, 0.22]

    aqi_txt, conf, tags = predict_from_news(sample_headlines, sample_embedding)
    print("AQI_txt:", aqi_txt)
    print("confidence:", conf)
    print("tags:", tags)
