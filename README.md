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
- [Contribution](#contribution)
- [License](#license)

## Overview
This repository contains a curated list of open-source tools for AI security, red teaming, and fairness assessment. These tools are categorized based on their functionality to help researchers, developers, and security professionals enhance AI systems' robustness, fairness, and security [4]. A good README file helps explain what the project is about, how users can install it, and how to use it [1].

---

## 1. Adversarial Machine Learning
Tools for generating adversarial examples and testing model defenses:
- **Adversarial Robustness Toolbox (ART)**: IBM's library for generating adversarial attacks and evaluating model robustness. ([Adversarial Robustness Toolbox (ART)](https://github.com/ibm/adversarial-robustness-toolbox))
- **Foolbox**: A Python library for creating adversarial examples to test neural network defenses. ([Foolbox](https://github.com/bethgelab/foolbox))
- **TextAttack**: A framework for adversarial attacks in NLP tasks. ([TextAttack](https://github.com/QData/TextAttack))
- **Counterfit**: Microsoft's CLI tool for automating adversarial AI red teaming. ([Counterfit](https://github.com/Azure/counterfit))
- **PyRIT**: The Python Risk Identification Tool for generative AI (PyRIT) is an open-source framework built to empower security professionals and engineers to proactively identify risks in generative AI systems.([PyRIT](https://github.com/Azure/PyRIT)

---

## 2. AI Model Extraction & Inference Attacks
Tools for reverse engineering models or testing privacy vulnerabilities:
- **Model Inversion Toolkit**: Reverse-engineers AI models using API responses. ([Model Inversion Toolkit](https://github.com/ffhibnese/Model-Inversion-Attack-ToolBox))
- **MIA (Membership Inference Attack)**: Tests whether specific data samples were used during model training. 
- **Pytorch Captum**: A library for model explainability and security auditing. ([Pytorch Captum](https://github.com/pytorch/captum))

---

## 3. AI Bias & Fairness Testing
Tools for assessing and mitigating bias in machine learning models:
- **AI Fairness 360 (AIF360)**: IBM's toolkit offering over 70 fairness metrics and 10 bias mitigation algorithms ([AI Fairness 360](https://ai-fairness-360.org))
- **Fairlearn**: Microsoft's library for fairness evaluation and mitigation in ML models. ([Fairlearn](https://github.com/fairlearn/fairlearn))

---

## 4. ML Model & Pipeline Security
Tools for securing machine learning pipelines:
- **SecML**: A library for security testing of ML models. ([SecML](https://github.com/secml/secml))
- **PrivacyRaven**: A framework for testing privacy vulnerabilities in AI systems. ([PrivacyRaven](https://github.com/AI-infrastructure-Foundation/PrivacyRaven))
- **Snorkel**: A data labeling and augmentation tool for detecting data poisoning. ([Snorkel](https://github.com/snorkel-ai/snorkel))

---

## 5. LLM Security & Red Teaming
Specialized tools for large language models (LLMs):
- **PyRIT (Python Risk Identification Toolkit)**: Microsoft's tool for automated red teaming of generative AI systems. It includes features like prompt generation, scoring engines, and attack strategies.
- **DeepEval**: An open-source framework designed for LLM evaluation and red teaming. ([DeepEval](https://github.com/confident-ai/deepeval))
- **Prompt Inject**: Evaluates vulnerabilities to prompt injection attacks. ([Prompt Inject](https://github.com/protectai/prompt-inject))
- **LLM Guard**: A security toolkit for filtering and mitigating LLM-based attacks. ([LLM Guard](https://github.com/laiyer-ai/llm-guard))

---

## 6. MLOps & Deployment Security
Tools focused on securing the deployment of AI models:
- **MLFlow**: A platform for model tracking with added security features. ([MLflow](https://github.com/mlflow/mlflow))
- **Trivy**: Scans containerized environments hosting AI applications for vulnerabilities. ([Trivy](https://github.com/aquasecurity/trivy))
- **Kubescape**: Ensures Kubernetes security in AI workloads. ([Kubescape](https://github.com/kubescape/kubescape))

---

## 7. AI Attack Simulation & Red Teaming Frameworks
Comprehensive frameworks for simulating AI attacks:
- **MITRE ATLAS**: MITRE's framework focuses on adversarial tactics in machine learning. ([MITRE ATLAS](https://atlas.mitre.org/))

---

## Contribution
We welcome contributions to this repository! Please submit a pull request or open an issue to suggest new tools or improvements. A good README should tell people how they can contribute to the project

## License
This repository is licensed under the [MIT License](LICENSE).
