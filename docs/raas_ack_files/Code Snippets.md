# RACEN — Code Snippets (Python/Pseudocode)

---

## 1. Ack Classifier (LLM few-shot)

```python
def classify_ack(user_message, prev_assistant_msg):
    prompt = f"""
Classification: ACK_CONTINUE or NEW_TOPIC.

Previous assistant: {prev_assistant_msg}
User: {user_message}

Examples:
Q: "where is grest head office?"
A: "address"
U: "yes please more info"
-> ACK_CONTINUE

U: "ok and your return policy?"
-> NEW_TOPIC

Now classify only with the token:
"""
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role":"user","content":prompt}]
    )
    label = resp.choices[0].message.content.strip()
    return label
```

---

## 2. Ack fallback classifier (cosine similarity)

```python
def ack_similarity(user_msg, prev_offer):
    score = cosine_sim(embed(user_msg), embed(prev_offer))
    return "ACK_CONTINUE" if score > 0.6 else "NEW_TOPIC"
```

---

## 3. Facet Selection

```python
def select_facet(intent, prev_facet, ack):
    bundle = load_json(f"config/facets/{intent}.json")

    if ack == "NEW_TOPIC":
        return None

    if prev_facet:
        # pick next facet in priority order
        pr = bundle["priority"]
        if prev_facet in pr:
            idx = pr.index(prev_facet)
            if idx + 1 < len(pr):
                return pr[idx+1]
        return pr[0]
    else:
        return bundle["priority"][0]
```

---

## 4. Retrieval-Bias

```python
def retrieve(query_vec, allowlist):
    docs = vector_db.search(query_vec, top_k=12)

    boosted = []
    for d in docs:
        if d.url in allowlist:
            d.score *= 1.2
        boosted.append(d)

    # dedupe near-duplicates
    clean = dedupe_docs(boosted)

    top_score = clean[0].score if clean else 0
    confidence = "HIGH" if top_score >= 0.78 else "LOW"

    return clean[:6], confidence
```

---

## 5. Prompt Builder

```python
def build_prompt(system_text, rules, history, facet, snippets, confidence, lang):
    return [
        {"role":"system","content":system_text},
        {"role":"system","content":rules},
        {"role":"system","content":f"language:{lang}"},
        {"role":"system","content":f"facet:{facet}"},
        {"role":"system","content":f"retrieval_confidence:{confidence}"},
        {"role":"system","content":f"snippets:{snippets}"},
        *history
    ]
```

---

## 6. Post-processing

```python
def postprocess(text, lang, tone):
    # enforce 2-sentence rule
    sentences = re.split(r'(?<=[.!?])\s+', text)
    primary = " ".join(sentences[:2])

    # tone adjustment (if needed)
    if tone == "empathetic":
        primary = soften(primary)

    return primary
```

