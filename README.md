# AI Red Teaming & Security Tools

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Last Verified](https://img.shields.io/badge/links_verified-2026--05-success.svg)
![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

A curated, practitioner-focused list of tools, frameworks, datasets, and resources for AI red teaming, adversarial ML, LLM security, and AI governance. Categorized by where each fits in a real workflow — not just by what it claims to do.

For a hands-on "where do I actually start" walkthrough mapped to an AI attack lifecycle, see [`TOOLING.md`](TOOLING.md).

> Links rot fast in this space. Each section is dated. Entries marked **⚠ deprecated** are kept for context — don't build on them.

---

## Table of Contents

- [How to Use This List](#how-to-use-this-list)
- [1. LLM Red Teaming & Scanning](#1-llm-red-teaming--scanning)
- [2. Prompt Injection & Jailbreak Research](#2-prompt-injection--jailbreak-research)
- [3. Agentic AI & MCP Attack Surface](#3-agentic-ai--mcp-attack-surface)
- [4. RAG & Vector Store Attacks](#4-rag--vector-store-attacks)
- [5. Multimodal Attacks](#5-multimodal-attacks)
- [6. Adversarial Machine Learning (Classical)](#6-adversarial-machine-learning-classical)
- [7. Model Extraction & Privacy Attacks](#7-model-extraction--privacy-attacks)
- [8. Data & Model Poisoning / Backdoors](#8-data--model-poisoning--backdoors)
- [9. Supply Chain & Model File Scanning](#9-supply-chain--model-file-scanning)
- [10. Guardrails & Runtime Defenses](#10-guardrails--runtime-defenses)
- [11. Evaluation Harnesses & Benchmarks](#11-evaluation-harnesses--benchmarks)
- [12. Bias, Fairness & Interpretability](#12-bias-fairness--interpretability)
- [13. MLOps / Deployment Security](#13-mlops--deployment-security)
- [14. Standards, Frameworks & Compliance](#14-standards-frameworks--compliance)
- [15. Incident & Vulnerability Databases](#15-incident--vulnerability-databases)
- [16. Playgrounds & CTFs](#16-playgrounds--ctfs)
- [17. Bug Bounty & Disclosure Programs](#17-bug-bounty--disclosure-programs)
- [18. Further Reading](#18-further-reading)
- [Contributing](#contributing)
- [License](#license)

---

## How to Use This List

If you're new to AI red teaming, a sensible starting path:

1. Read **OWASP LLM Top 10 (2025)** and **MITRE ATLAS** to get the threat model.
2. Run **garak** against a target LLM to see what automated scanning gets you.
3. Try **PyRIT** for orchestrated multi-turn attacks.
4. Walk through **Gandalf** (or any CTF in section 16) for hands-on prompt injection.
5. For applied work, read **NIST AI 600-1** and **NIST AI 100-2** to align findings to risk language stakeholders understand.

For deeper workflow guidance, see [`TOOLING.md`](TOOLING.md).

---

## 1. LLM Red Teaming & Scanning

Automated and semi-automated tooling for testing LLM systems.

- **[PyRIT](https://github.com/Azure/PyRIT)** — Microsoft's Python Risk Identification Tool. Orchestrated, multi-turn red teaming for generative AI. Probably the most full-featured open-source framework right now.
- **[garak](https://github.com/NVIDIA/garak)** — NVIDIA's LLM vulnerability scanner. Probe-and-detector architecture, dozens of attack categories (encoding tricks, DAN variants, prompt leak, toxicity, RealToxicityPrompts, etc.). Think `nmap` for LLMs.
- **[promptfoo](https://github.com/promptfoo/promptfoo)** — LLM evaluation + red teaming. Strong YAML-driven test harness, OWASP LLM Top 10 preset built in.
- **[DeepEval](https://github.com/confident-ai/deepeval)** — pytest-style LLM evaluation framework with red team modules.
- **[Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)** — UK AISI's evaluation framework. Increasingly the standard for safety evals and dangerous-capability testing.
- **[Mantis](https://github.com/PasqualeDeRosa/mantis)** *(verify before use — research project, lightly maintained)* — Trail of Bits' framework for LLM adversarial testing
- **[TextAttack](https://github.com/QData/TextAttack)** — Adversarial attacks on NLP models (still useful for classifier-style targets; less relevant to large autoregressive models).
- ⚠ **Counterfit** — Microsoft archived this in 2023 in favor of PyRIT. Don't start here.

---

## 2. Prompt Injection & Jailbreak Research

Tools, payload collections, and research benches for prompt injection and jailbreaking specifically.

- **[Promptmap2](https://github.com/utkusen/promptmap)** — Automated prompt injection scanner for LLM apps.
- **[L1B3RT4S](https://github.com/elder-plinius/L1B3RT4S)** — Plinius the Liberator's jailbreak prompt collection. Updated frequently; widely used as a corpus.
- **[Many-shot jailbreaking](https://www.anthropic.com/research/many-shot-jailbreaking)** — Anthropic research on long-context jailbreaks
- **[Crescendo](https://crescendo-the-multiturn-jailbreak.github.io/)** — Microsoft Research multi-turn jailbreak technique
- **[Skeleton Key](https://www.microsoft.com/en-us/security/blog/2024/06/26/mitigating-skeleton-key-a-new-type-of-generative-ai-jailbreak-technique/)** — Microsoft-disclosed jailbreak class
- **[JailbreakBench](https://github.com/JailbreakBench/jailbreakbench)** — standardized jailbreak benchmark
- **[HarmBench](https://github.com/centerforaisafety/HarmBench)** — automated red teaming benchmark from CAIS
- **[AdvBench](https://github.com/llm-attacks/llm-attacks)** — GCG-style suffix attacks; companion to the original Zou et al. "universal and transferable adversarial attacks" paper
- **[BurpGPT](https://github.com/aress31/burpgpt)** — Burp Suite extension that integrates LLM analysis into web testing flows

---

## 3. Agentic AI & MCP Attack Surface

Tools and benches for agent systems and Model Context Protocol — the hot 2025–2026 attack surface.

- **[AgentDojo](https://github.com/ethz-spylab/agentdojo)** — Benchmark for agent prompt injection attacks (ETH Zurich)
- **[InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent)** — Indirect prompt injection benchmark for tool-using agents
- **[Agent Security Bench (ASB)](https://github.com/agiresearch/ASB)** — Formal benchmark for LLM agent security
- **[τ-bench](https://github.com/sierra-research/tau-bench)** — Tool-agent-user benchmark (Sierra)
- **[mcp-scan](https://github.com/invariantlabs-ai/mcp-scan)** *(verify — fast-moving space)* — Invariant Labs' scanner for MCP server risks
- **[OWASP Top 10 for Agentic AI](https://genai.owasp.org/resource/agentic-ai-threats-and-mitigations/)** *(2025)* — threats and mitigations document
- **[NIST AI RMF Agentic Profile](https://labs.cloudsecurityalliance.org/agentic/agentic-nist-ai-rmf-profile-v1/)** *(draft, CSA Labs)* — extension to AI RMF for autonomous agents

---

## 4. RAG & Vector Store Attacks

- **[PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG)** — Knowledge corruption attacks against RAG pipelines
- **[AgentPoison](https://github.com/BillChan226/AgentPoison)** — Memory-poisoning attacks on agent RAG memory
- **[ConfusedPilot](https://confusedpilot.info/)** — RAG-based Copilot attack class
- See also: section 1 tools (garak, promptfoo) have RAG-specific probes/presets

---

## 5. Multimodal Attacks

- **[MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench)** — Vision-language model safety benchmark
- **[VLAttack](https://github.com/ericyinyzy/VLAttack)** — Visual-language adversarial attacks
- **[Voice Jailbreak Attacks](https://github.com/TrustAIRLab/VoiceJailbreakAttacks)** — Audio modality attacks
- **[FigStep](https://github.com/ThuCCSLab/FigStep)** — Image-encoded prompt injection
- **Adversarial patch / image attacks** — see [ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox) in section 6 for the classical toolkit

---

## 6. Adversarial Machine Learning (Classical)

Pre-LLM-era adversarial example tooling. Still relevant for classifiers, vision, and recommender models.

- **[Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)** — IBM/LF AI; the most comprehensive adversarial ML library. 39+ attacks, 29+ defenses.
- **[Foolbox](https://github.com/bethgelab/foolbox)** — Adversarial examples for PyTorch/TensorFlow/JAX models
- **[CleverHans](https://github.com/cleverhans-lab/cleverhans)** — Benchmarking adversarial defenses (note: moved to `cleverhans-lab` org)
- **[SecML](https://github.com/pralab/secml)** — ML security against adversarial / poisoning / evasion attacks

---

## 7. Model Extraction & Privacy Attacks

- **[ML Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter)** — Membership inference, attribute inference, model inversion
- **[Model Inversion Attack Toolbox](https://github.com/ffhibnese/Model-Inversion-Attack-ToolBox)** — Reconstruct training data from model outputs
- **[TensorFlow Privacy](https://github.com/tensorflow/privacy)** — Differential-privacy training + attack benchmarks
- **[Opacus](https://github.com/pytorch/opacus)** — DP-SGD for PyTorch
- **Training data extraction research** — see [Carlini et al. on extractable training data](https://arxiv.org/abs/2311.17035) for current technique baselines

---

## 8. Data & Model Poisoning / Backdoors

- **[BackdoorBench](https://github.com/SCLBD/BackdoorBench)** — Comprehensive backdoor attack/defense benchmark
- **[TrojanZoo](https://github.com/ain-soph/trojanzoo)** — Backdoor and adversarial robustness library
- **[BadNets](https://github.com/Kooscii/BadNets)** — Classic backdoor reference implementation
- **Snorkel** *(formerly recommended for "poisoning detection" — that framing was loose)* — `snorkel.org`; data labeling tool, not strictly a security tool

---

## 9. Supply Chain & Model File Scanning

The pickle problem hasn't gone away; safetensors has, but compromised models still ship.

- **[ModelScan](https://github.com/protectai/modelscan)** — ProtectAI scanner for ML model serialization formats (pickle, H5, SavedModel, GGUF, etc.)
- **[picklescan](https://github.com/mmaitre314/picklescan)** — Static analysis for malicious pickle files
- **[Fickling](https://github.com/trailofbits/fickling)** — Trail of Bits' pickle decompiler and security analyzer
- **[HuggingFace Picklescan integration](https://huggingface.co/docs/hub/security-pickle)** — built-in scanning on the Hub
- **[Stable Diffusion Pickle Scanner](https://github.com/zxix/stable-diffusion-pickle-scanner)** + [GUI](https://github.com/diStyApps/Stable-Diffusion-Pickle-Scanner-GUI) — SD ecosystem-specific
- **[Guardian](https://protectai.com/guardian)** *(commercial)* — ProtectAI's enterprise model scanning
- **[HiddenLayer Model Scanner](https://hiddenlayer.com/)** *(commercial)* — model file scanning + attack telemetry

---

## 10. Guardrails & Runtime Defenses

For builders, but red teamers should know what they're testing against.

- **[LLM Guard](https://github.com/protectai/llm-guard)** — ProtectAI's input/output scanner toolkit (was `laiyer-ai/llm-guard` — repo moved). 15 input + 20 output scanners.
- **[NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)** — NVIDIA's programmable guardrails (Colang DSL)
- **[Rebuff](https://github.com/protectai/rebuff)** — Prompt injection detection (heuristics + LLM classifier + vector DB + canary tokens)
- **[Guardrails AI](https://github.com/guardrails-ai/guardrails)** — Validation layer for LLM I/O
- **[Vigil](https://github.com/deadbits/vigil-llm)** — LLM prompt injection scanner
- **[LangKit](https://github.com/whylabs/langkit)** — WhyLabs' LLM telemetry / monitoring with safety signals
- **[Lakera Guard](https://www.lakera.ai/lakera-guard)** *(commercial, free tier)* — hosted prompt injection / data loss API
- **[Llama Guard 3 / 4](https://github.com/meta-llama/PurpleLlama)** — Meta's safety classifier models (open weights)

---

## 11. Evaluation Harnesses & Benchmarks

Red teaming and eval converge. Use these to baseline a model and measure post-mitigation deltas.

- **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** — EleutherAI's standard eval harness (200+ tasks)
- **[HELM](https://github.com/stanford-crfm/helm)** — Stanford CRFM's holistic evaluation
- **[OpenAI Evals](https://github.com/openai/evals)** — Eval framework + registry
- **[Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)** — UK AISI; cross-listed with section 1
- **[CyberSecEval](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks)** — Meta's cyber-risk benchmark for LLMs (insecure code, cyberattack helpfulness, etc.)
- **[METR Task Suite](https://github.com/METR/task-standard)** — autonomy / dangerous-capability evals
- **[SWE-bench](https://github.com/SWE-bench/SWE-bench)** — code-agent benchmark (relevant for evaluating code-writing capability that matters in security contexts)

---

## 12. Bias, Fairness & Interpretability

- **[AI Fairness 360 (AIF360)](https://github.com/Trusted-AI/AIF360)** — IBM; 70+ fairness metrics, 10 bias mitigation algorithms
- **[Fairlearn](https://github.com/fairlearn/fairlearn)** — Microsoft fairness assessment + mitigation
- **[Captum](https://github.com/pytorch/captum)** — PyTorch model interpretability
- **[SHAP](https://github.com/shap/shap)** — Shapley value explanations
- **[LIME](https://github.com/marcotcr/lime)** — Local interpretable model-agnostic explanations
- **[Transformer Circuits](https://transformer-circuits.pub/)** — Anthropic's interpretability research (reference, not tooling)

---

## 13. MLOps / Deployment Security

- **[MLflow](https://github.com/mlflow/mlflow)** — Experiment tracking; check security advisories, several CVEs over 2023–2024
- **[Trivy](https://github.com/aquasecurity/trivy)** — Container & filesystem scanner; works on AI containers too
- **[Kubescape](https://github.com/kubescape/kubescape)** — Kubernetes hardening
- **[NB Defense](https://github.com/protectai/nbdefense)** — Jupyter notebook security scanner
- **[Morpheus](https://github.com/nv-morpheus/Morpheus)** — NVIDIA cybersecurity AI pipeline (anomaly detection, etc.) — note: defensive/SOC framing, not red team

---

## 14. Standards, Frameworks & Compliance

**Cross-cutting frameworks**
- **[MITRE ATLAS](https://atlas.mitre.org/)** — Adversarial threat landscape for AI systems. Counterpart to ATT&CK; the threat model most red teamers map findings to.
- **[OWASP Top 10 for LLM Applications (v2025)](https://genai.owasp.org/llm-top-10/)** — Current version; covers prompt injection, supply chain, system prompt leakage, vector/embedding weaknesses, unbounded consumption, etc.
- **[OWASP Top 10 for Agentic AI (2025)](https://genai.owasp.org/resource/agentic-ai-threats-and-mitigations/)** — Agent-specific risk taxonomy
- **[OWASP ML Security Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)** — Classical ML risks
- **[OWASP AI Exchange](https://owaspai.org/)** — Comprehensive AI security & governance reference

**NIST**
- **[NIST AI RMF 1.0 (AI 100-1)](https://www.nist.gov/itl/ai-risk-management-framework)** — Core AI risk management framework
- **[NIST AI 600-1](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)** — Generative AI Profile of the AI RMF (July 2024). 12 risk categories, 200+ suggested actions.
- **[NIST AI 100-2 E2025](https://csrc.nist.gov/pubs/ai/100/2/e2025/final)** — Adversarial Machine Learning: A Taxonomy and Terminology
- **[NIST SP 800-218A](https://csrc.nist.gov/pubs/sp/800/218/a/final)** — Secure Software Development Practices for Generative AI / Dual-Use Foundation Models
- **[Dioptra](https://github.com/usnistgov/dioptra)** — NIST's testbed for assessing ML attack effects

**International / Regulatory**
- **[EU AI Act](https://artificialintelligenceact.eu/)** — In force progressively from Feb 2025; risk-tiered obligations (unacceptable / high / limited / minimal)
- **[ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html)** — AI Management Systems standard
- **[ISO/IEC 23894:2023](https://www.iso.org/standard/77304.html)** — AI risk management
- **[ISO/IEC 27090](https://www.iso.org/standard/56581.html)** *(in development)* — Guidance on addressing security threats to AI systems
- **[UK AI Safety Institute](https://www.aisi.gov.uk/)** — Publications and frameworks
- **[Singapore Model AI Governance Framework](https://www.pdpc.gov.sg/help-and-resources/2020/01/model-ai-governance-framework)**
- **[GDPR & AI](https://gdpr.eu/ai-and-gdpr/)** — Article 22 (automated decisions), and broader DPIA expectations
- **[Korea AI Basic Act (2025)](https://www.korea.kr/news/policyNewsView.do?newsId=148937049)** *(verify current URL — recent legislation)*

**Vendor & Industry**
- **[Google Secure AI Framework (SAIF)](https://safety.google/cybersecurity-advancements/saif/)**
- **[Microsoft Responsible AI Standard](https://www.microsoft.com/en-us/ai/responsible-ai)**
- **[Anthropic's Responsible Scaling Policy](https://www.anthropic.com/news/anthropics-responsible-scaling-policy)**
- **[Frontier Model Forum](https://www.frontiermodelforum.org/)** — Industry body output (Anthropic, Google, Microsoft, OpenAI)

> Note on US executive actions: EO 14110 (Biden, 2023) was revoked in January 2025 and replaced by Executive Order 14179 ("Removing Barriers to American Leadership in Artificial Intelligence"). NIST AI 600-1 was published *under* EO 14110 but the document itself remains a NIST publication and is still in use. The US executive landscape is shifting; verify current EO and OMB guidance before relying on either.

---

## 15. Incident & Vulnerability Databases

- **[AI Vulnerability Database (AVID)](https://avidml.org/)** — Open knowledge base of failure modes in GPAI systems; pivoted in 2025–2026 to focus on agentic / system-level vulns. Maps to MITRE ATLAS, CVSS, and AVID's own taxonomy.
- **[AI Incident Database (AIID)](https://incidentdatabase.ai/)** — Real-world AI incidents (Partnership on AI)
- **[OECD AI Incidents Monitor](https://oecd.ai/en/incidents)** — Tracks AI-related incidents internationally
- **[MITRE ATLAS Case Studies](https://atlas.mitre.org/studies)** — Real attacks mapped to the ATLAS matrix

---

## 16. Playgrounds & CTFs

For practice, training, and demonstrating attacks safely.

- **[Gandalf](https://gandalf.lakera.ai/)** — Lakera's prompt-injection CTF, 8 levels + extras. The classic starter.
- **[Tensor Trust](https://tensortrust.ai/)** — Prompt injection / defense game (Berkeley)
- **[Doublespeak](https://doublespeak.chat/)** — Jailbreak chat-style challenges
- **[Spy Logic](https://www.immersivelabs.com/resources/blog/prompt-injection-attacks-on-applications-that-use-llms)** — Immersive Labs LLM challenges
- **[PortSwigger Web Security Academy — LLM labs](https://portswigger.net/web-security/llm-attacks)** — Practical hands-on labs for LLM-integrated web apps
- **[DEF CON AI Village Generative Red Team](https://aivillage.org/)** — Annual large-scale public red team event
- **[HackTheBox AI-themed boxes](https://www.hackthebox.com/)** — Various AI-themed challenges across HTB and HTB Academy
- **[Prompt Airlines](https://promptairlines.com/)** — Wiz's prompt injection CTF
- **[MyLLMBank](https://myllmbank.com/)** — LLM stress-test playground
- **[Hugging Face Spaces](https://huggingface.co/spaces)** — Host of community AI demos; many double as test targets

---

## 17. Bug Bounty & Disclosure Programs

- **[0Din by Mozilla](https://0din.ai/)** — Generative AI–focused bug bounty
- **[Anthropic Responsible Disclosure / Bug Bounty](https://hackerone.com/anthropic)** — Public program covering Claude
- **[OpenAI Bug Bounty](https://openai.com/security/bug-bounty-program/)** — Covers OpenAI products/infra; note model-output issues have a separate process
- **[Google VRP (AI scope)](https://bughunters.google.com/about/rules/google-friends/6625378258649088/google-and-alphabet-vulnerability-reward-program-vrp-rules)** — AI vulnerabilities included in main VRP
- **[Microsoft AI Bounty](https://www.microsoft.com/en-us/msrc/bounty-ai)** — Covers Copilot and AI products
- **[Meta Bug Bounty](https://www.facebook.com/whitehat/)** — AI/Llama scope expanded 2024+
- **[Hugging Face Security](https://huggingface.co/docs/hub/security)** *(disclosure, not paid bounty by default)*
- **[xAI Bug Bounty](https://hackerone.com/x)** *(verify current scope)* — Grok and related
- **[Bugcrowd AI Security](https://www.bugcrowd.com/solutions/ai-security/)** — Platform with multiple AI-targeted programs
- **[HackerOne AI Safety](https://www.hackerone.com/ai-safety)** — Platform-hosted programs and challenges
- **[Apple Security Bounty (AI scope)](https://security.apple.com/bounty/categories/)** — Apple Intelligence is now in scope

---

## 18. Further Reading

- **[Lilian Weng — Adversarial Attacks on LLMs](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/)** — Survey-quality overview
- **[Simon Willison's prompt injection writing](https://simonwillison.net/tags/promptinjection/)** — The single most prolific public commentator on the topic; start here
- **[Anthropic's red teaming research](https://www.anthropic.com/research)** — Particularly the constitutional classifiers and many-shot jailbreaking papers
- **[OpenAI Safety Research](https://openai.com/safety/)**
- **[Google DeepMind safety publications](https://deepmind.google/discover/blog/)**
- **[NVIDIA AI Red Team blog](https://developer.nvidia.com/blog/tag/ai-security/)**
- **[AI Snake Oil](https://www.aisnakeoil.com/)** — Useful counterweight to hype; critical thinking on AI claims
- **[Embrace the Red (Johann Rehberger)](https://embracethered.com/blog/)** — Practical prompt injection / agent exploit research

---

## Contributing

PRs welcome. Useful contributions:
- New tools that meaningfully change a workflow (not just another wrapper)
- Replacements for deprecated tools
- Updates to standards / frameworks
- Broken-link fixes (please verify before submitting)

Please keep entries tight, cite primary sources, and note if something is **research-quality** vs **production-ready** vs **commercial**.

---

## License

[MIT](LICENSE) — use freely, attribution appreciated.

---

This is a living document. The AI security field moves quickly; expect quarterly-ish refreshes.
