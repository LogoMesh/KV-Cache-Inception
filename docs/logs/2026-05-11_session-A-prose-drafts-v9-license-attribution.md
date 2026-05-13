# Session A v9 Prose Draft — Llama 3.2 License Attribution

**Authored:** 2026-05-11 (Day 3 of ARR-cycle execution plan v2)
**Audience:** Session B (TeX application target — §7.6 Ethical Considerations)
**Status:** Day-3 first draft. **Verbatim attribution text confirmed via primary source** ([https://www.llama.com/llama3_2/license/](https://www.llama.com/llama3_2/license/), fetched 2026-05-11).

---

## Primary-source verbatim text (Josh J2)

Three binding requirements from the Llama 3.2 Community License:

1. **"Built with Llama"** must be prominently displayed (website, UI, blogpost, about page, or product documentation).
2. **Notice file required** with the exact attribution string:
   > Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
3. **Acceptable Use Policy** must be referenced: https://llama.com/llama3_2/use-policy/

The license does not specify particular model size variants — covers all of Llama 3.2 generically.

---

## Replacement TeX for §7.6 Ethical Considerations

**TeX target:** Canonical TeX line 588 (current §7.6 Ethical Considerations body — single paragraph).
**Insertion strategy:** Append a NEW paragraph after the existing single paragraph (preserve Croissant 1.1 sentence). Do not replace.

**Replacement body** (appended at end of §7.6, before `% [REVISION | §8-cut]` marker):

```latex
\paragraph{Llama 3.2 license attribution.}
The Reversible KV-Cache MCTS algorithm reported in this work is evaluated on Meta's Llama~3.2-1B-Instruct and Llama~3.2-3B-Instruct models. Built with Llama. Llama~3.2 is licensed under the Llama 3.2 Community License, Copyright \textcopyright{} Meta Platforms, Inc. All Rights Reserved. Our experimental use complies with the Llama 3.2 Acceptable Use Policy~\citep{meta2024llama3_2_aup}.
```

**Bibliography entry to add** (in canonical TeX bibliography block):

```latex
\bibitem[Meta(2024)]{meta2024llama3_2_aup}
Meta Platforms, Inc.
\newblock {Llama 3.2 Acceptable Use Policy}.
\newblock \url{https://llama.com/llama3_2/use-policy/}, 2024.
```

---

## Notes for Session B

**REVISION marker for this addition:**

```latex
% [REVISION | §7.6-Llama-license-attribution-D+ | 2026-05-11 | Added Llama 3.2 license attribution paragraph per Josh J2 (verbatim from https://www.llama.com/llama3_2/license/ fetched 2026-05-11). Three license-required elements present: (a) "Built with Llama" prominent display; (b) verbatim notice text per Llama 3.2 Community License § "Notice"; (c) acceptable-use-policy URL reference. The "Built with Llama" phrase is intentionally appended after the model-name sentence to satisfy the "prominently displayed" requirement in §Ethical Considerations of a paper. New \cite{meta2024llama3_2_aup} bibliography entry added; URL points to canonical Meta-hosted policy page (no archived/cached version since rules can shift before camera-ready — verify URL at submission-time per plan v2 §6 J2 follow-up). — pending audit]
```

**Compilation impact:**
- Net +1 paragraph in §7.6 (~50 words)
- Net +1 bibliography entry
- `\citep{meta2024llama3_2_aup}` resolves via the new bibitem
- `\textcopyright{}` is in LaTeX's standard text-mode

**Cross-reference resolution:**
- No new labels added
- No forward-references introduced

**G-license gate (plan v2 §9):** Day 10 verifies that the verbatim attribution string `Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.` appears verbatim in the compiled PDF. This draft satisfies that gate.

---

## Acceptable Use Policy compliance check (not in TeX, internal record)

The Llama 3.2 AUP at https://llama.com/llama3_2/use-policy/ (referenced for our compliance, not required in paper text):
- Use case: latent-space interpretability research → permitted (research is a generally-permitted category)
- No deployment to end users → no AUP risk
- No commercial product → no AUP risk
- Adversarial probing within own research → covered under research category (not "use to defeat or evade safety mitigations" which is restricted)

Compliance affirmation: this work uses Llama 3.2 for offline research analysis of model internals; outputs are not deployed; experimental probes operate on the model's own representations rather than as inputs from external users. Compatible with the Llama 3.2 AUP.

---

*End of license-attribution draft. Apply Day 3 P5 (TeX edit + bibitem addition). G-license gate (Day 10) verifies presence in compiled PDF.*
