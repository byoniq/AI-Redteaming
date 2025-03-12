# AI Red Teaming and Security Tools Repository

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Table of Contents
- [Overview](#overview)
- [1. Adversarial Machine Learning](#1-adversarial-machine-learning)
- [2. AI Model Extraction & Inference Attacks](#2-ai-model-extraction--inference-attacks)
- [3. AI Bias & Fairness Testing](#3-ai-bias--fairness-testing)
- [4. ML Model & Pipeline Security](#4-ml-model--pipeline-security)
- [5. LLM Security & Red Teaming](#5-llm-security--red-teaming)
- [6. MLOps & Deployment Security](#6-mlops--deployment-security)
- [7. AI Attack Simulation & Red Teaming Frameworks](#7-ai-attack-simulation--red-teaming-frameworks)
- [8. Standards, Frameworks, & Compliance](#8-standards-frameworks--compliance)
- [9. Incident Databases](#9-incident-databases)
- [10. Python Pickle Scanners](#10-python-pickle-scanners)
- [11. AI Red Teaming Playgrounds](#11-ai-red-teaming-playgrounds)
- [12. AI Bug Bounty Programs](#12-ai-bug-bounty-programs)
- [Contribution](#contribution)
- [License](#license)

## Overview
This repository curates open-source tools, frameworks, and resources for AI security, red teaming, and fairness assessment. Designed for researchers, developers, and security professionals, it categorizes tools by functionality to enhance AI system robustness, fairness, and security. Contributions are welcome to keep this resource comprehensive and up-to-date.

---

## 1. Adversarial Machine Learning
Tools for generating adversarial examples and testing model defenses:
- **Adversarial Robustness Toolbox (ART)**: IBM’s library for crafting adversarial attacks and evaluating model robustness. ([GitHub](https://github.com/Trusted-AI/adversarial-robustness-toolbox))
- **Foolbox**: A Python library to create adversarial examples for testing neural network resilience. ([GitHub](https://github.com/bethgelab/foolbox))
- **TextAttack**: A framework for adversarial attacks on NLP models. ([GitHub](https://github.com/QData/TextAttack))
- **Counterfit**: Microsoft’s CLI tool to automate adversarial AI red teaming. ([GitHub](https://github.com/Azure/counterfit))
- **PyRIT**: Microsoft’s Python Risk Identification Tool for generative AI risk assessment. ([GitHub](https://github.com/Azure/PyRIT)) ([YouTube Talk](https://www.youtube.com/watch?v=your-link-here)) *Note: Replace with actual YouTube link if available.*
- **CleverHans**: A library for benchmarking adversarial example defenses. ([GitHub](https://github.com/cleverhans-team/cleverhans))
- **Promptfoo**: A tool for testing LLM robustness via adversarial prompts. ([GitHub](https://github.com/promptfoo/promptfoo))

---

## 2. AI Model Extraction & Inference Attacks
Tools for reverse-engineering models or testing privacy vulnerabilities:
- **Model Inversion Toolkit**: Reverse-engineers AI models using API responses. ([GitHub](https://github.com/ffhibnese/Model-Inversion-Attack-ToolBox))
- **MIA (Membership Inference Attack)**: Tests if data samples were used in training. ([GitHub](https://github.com/privacytrustlab/ml_privacy_meter))
- **Pytorch Captum**: A library for model interpretability and security auditing. ([GitHub](https://github.com/pytorch/captum))
- **Nvidia Morpheus**: A framework for cybersecurity in AI pipelines, including inference attack detection. ([GitHub](https://github.com/nvidia/morpheus))
- **Nvidia Garak**: A scanner for vulnerabilities in generative AI models. ([GitHub](https://github.com/leondz/garak))
- **BurpGPT**: Integrates LLM security testing with Burp Suite. ([GitHub](https://github.com/aress31/burpgpt))
- **GraphRAG**: A retrieval-augmented generation tool with potential security testing applications. ([GitHub](https://github.com/microsoft/graphrag))

---

## 3. AI Bias & Fairness Testing
Tools for assessing and mitigating bias in machine learning models:
- **AI Fairness 360 (AIF360)**: IBM’s toolkit with 70+ fairness metrics and 10 bias mitigation algorithms. ([Website](https://aif360.res.ibm.com/)) ([GitHub](https://github.com/Trusted-AI/AIF360))
- **Fairlearn**: Microsoft’s library for fairness evaluation and mitigation. ([GitHub](https://github.com/fairlearn/fairlearn))

---

## 4. ML Model & Pipeline Security
Tools for securing machine learning models and pipelines:
- **SecML**: A library for security testing of ML models against adversarial threats. ([GitHub](https://github.com/pralab/secml))
- **PrivacyRaven**: Tests privacy vulnerabilities in AI systems. ([GitHub](https://github.com/AI-infrastructure-Foundation/PrivacyRaven))
- **Snorkel**: A data labeling tool that can detect data poisoning risks. ([GitHub](https://github.com/snorkel-team/snorkel))
- **TextAttack**: Also applicable here for NLP model security testing. ([GitHub](https://github.com/QData/TextAttack))

---

## 5. LLM Security & Red Teaming
Specialized tools for large language models (LLMs):
- **PyRIT**: Microsoft’s automated red teaming tool for generative AI. ([GitHub](https://github.com/Azure/PyRIT))
- **DeepEval**: A framework for LLM evaluation and red teaming. ([GitHub](https://github.com/confident-ai/deepeval))
- **Prompt Inject**: Tests vulnerabilities to prompt injection attacks. ([GitHub](https://github.com/protectai/prompt-inject))
- **LLM Guard**: Filters and mitigates LLM-based attacks. ([GitHub](https://github.com/laiyer-ai/llm-guard))
- **LM-Studio**: A platform for running and testing LLMs locally with security implications. ([GitHub](https://github.com/lmstudio-ai/lm-studio))
- **Octopii**: An AI-powered scanner for detecting PII in LLM outputs. ([GitHub](https://github.com/redhuntlabs/Octopii))

---

## 6. MLOps & Deployment Security
Tools for securing AI model deployment:
- **MLFlow**: Tracks models with security features for MLOps workflows. ([GitHub](https://github.com/mlflow/mlflow))
- **Trivy**: Scans containerized AI environments for vulnerabilities. ([GitHub](https://github.com/aquasecurity/trivy))
- **Kubescape**: Ensures Kubernetes security for AI workloads. ([GitHub](https://github.com/kubescape/kubescape))

---

## 7. AI Attack Simulation & Red Teaming Frameworks
Comprehensive frameworks for simulating AI attacks:
- **MITRE ATLAS**: A knowledge base of adversarial tactics in ML systems. ([Website](https://atlas.mitre.org/))

---

## 8. Standards, Frameworks, & Compliance
Resources for AI governance, risk management, and compliance:
- **EU Artificial Intelligence Act**: Legislation governing AI use in the EU. ([Link](https://artificialintelligenceact.eu/))
- **Artificial Intelligence Risk & Compliance (AIRS)**: A framework for AI risk management. ([Link](https://airsinstitute.org/)) *Note: Placeholder; verify exact URL.*
- **ISO/IEC 42001 - AI Management Systems**: Standard for managing AI systems. ([Link](https://www.iso.org/standard/81230.html))
- **NIST AI RMF**: Risk management framework for AI from NIST. ([Link](https://www.nist.gov/itl/ai-risk-management-framework))
- **Blueprint for an AI Bill of Rights (US)**: Guidelines for ethical AI in the US. ([Link](https://www.whitehouse.gov/ostp/ai-bill-of-rights/))
- **GDPR Impact on AI**: How GDPR affects AI data practices. ([Link](https://gdpr.eu/ai-and-gdpr/))
- **Google Secure AI Framework (SAIF)**: Google’s approach to secure AI development. ([Link](https://cloud.google.com/security/ai-framework))
- **OWASP LLM Top 10**: Risks and mitigations for LLMs. ([Link](https://genai.owasp.org/llm-top-10/))
- **OWASP Machine Learning Security Top 10**: Security risks in ML systems. ([Link](https://owasp.org/www-project-machine-learning-security-top-10/))

---

## 9. Incident Databases
Repositories of AI-related security incidents:
- **Artificial Incidents Global Security Incidents Database**: Tracks AI security incidents worldwide. ([Link](https://incidentdatabase.ai/))

---

## 10. Python Pickle Scanners
Tools for detecting vulnerabilities in Python pickle files:
- **Picklescan**: Scans pickle files for malicious code. ([GitHub](https://github.com/mmaitre314/picklescan))
- **ProtectAI Modelscan**: Scans ML models for security risks, including pickle exploits. ([GitHub](https://github.com/protectai/modelscan))
- **Stable Diffusion Pickle Scanner**: Detects threats in Stable Diffusion pickle files. ([GitHub](https://github.com/zxix/stable-diffusion-pickle-scanner))
- **Stable Diffusion Pickle Scanner GUI**: A GUI version of the scanner. ([GitHub](https://github.com/diStyApps/Stable-Diffusion-Pickle-Scanner-GUI))
- **Fickling**: A tool to detect and mitigate pickle file attacks. ([GitHub](https://github.com/trailofbits/fickling))

---

## 11. AI Red Teaming Playgrounds
Interactive platforms for experimenting with and red teaming AI models, particularly LLMs:
- **MyLLMBank**: A playground for testing and stressing LLMs with various prompts to evaluate robustness and vulnerabilities. ([Website](https://myllmbank.com/))
- **Vercel AI Playground**: Allows side-by-side comparison of LLM responses for testing and red teaming purposes. ([Website](https://play.vercel.ai/))
- **AssemblyAI Playground**: An environment to test speech-to-text models and explore LLM interactions with audio data. ([Website](https://www.assemblyai.com/playground))
- **PromptSandbox.io**: A platform for prototyping and refining AI models with pre-trained GPTs, useful for red teaming experiments. ([Website](https://promptsandbox.io/))
- **Hugging Face Spaces**: Hosts interactive demos and playgrounds for testing LLMs and other AI models, often with red teaming potential. ([Website](https://huggingface.co/spaces))

---

## 12. AI Bug Bounty Programs
Programs incentivizing researchers to identify and report vulnerabilities in AI systems:
- **0Din by Mozilla**: A pioneering GenAI bug bounty program targeting vulnerabilities in LLMs and deep learning systems, offering rewards from $500 to $15,000 based on severity. ([Website](https://0din.ai/))
- **Bugcrowd AI Security Program**: A crowdsourced platform hosting bug bounties for AI-related vulnerabilities, often partnered with companies deploying AI solutions. ([Website](https://www.bugcrowd.com/solutions/ai-security/))
- **HackerOne AI Bug Bounty Challenges**: Offers specific challenges for AI and ML vulnerabilities, collaborating with organizations to secure AI deployments. ([Website](https://www.hackerone.com/vulnerability-management/ai-security))
- **OpenAI Bug Bounty Program**: Rewards researchers for finding security flaws in OpenAI’s systems, including ChatGPT and its underlying infrastructure. ([Website](https://openai.com/security/bug-bounty-program))
- **Google Vulnerability Reward Program (VRP) for AI**: Expanded to include AI-specific vulnerabilities in Google’s ML and generative AI products. ([Website](https://bughunters.google.com/about/rules))

---

## Contribution
Contributions are encouraged! Submit a pull request or open an issue to suggest new tools, updates, or improvements.

## License
This repository is licensed under the [MIT License](LICENSE).
