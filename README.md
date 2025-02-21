# A Robust Semantics-based Watermark for Large Language Model against Paraphrasing

Official code for [A Robust Semantics-based Watermark for Large Language Model against Paraphrasing](https://aclanthology.org/2024.findings-naacl.40/)

Large language models (LLMs) have show great ability in various natural language tasks. However, there are concerns that LLMs are possible to be used improperly or even illegally. To prevent the malicious usage of LLMs, detecting LLM-generated text becomes crucial in the deployment of LLM applications. Watermarking is an effective strategy to detect the LLM-generated content by encoding a pre-defined secret watermark to facilitate the detection process. However, the majority of existing watermark methods leverage the simple hashes of precedent tokens to partition vocabulary. Such watermark can be easily eliminated by paraphrase and correspondingly the detection effectiveness will be greatly compromised. Thus, to enhance the robustness against paraphrase, we propose a semantics-based watermark framework SemaMark. It leverages the semantics as an alternative to simple hashes of tokens since the paraphrase will likely preserve the semantic meaning of the sentences. Comprehensive experiments are conducted to demonstrate the effectiveness and robustness of SemaMark under different paraphrases.

## Prelimnaries

First of all, we need to replace some codes in transformers lib. Please replace with the class and functions in replace_transformer.py

## How to use

### Train

The ring is trained by **run_pipeline.sh**

The checkpiont for OPT-2.7B is [Google drive](https://drive.google.com/file/d/1oCM88Gx0kG1K0mLv5bgWk210-TWU8rA9/view?usp=sharing)

### Watermarking

Detailed CMD of watermarking can be found in **generation.sh**

### Acknowledgement

This is code is based on [A Watermark for Large language Models](https://github.com/jwkirchenbauer/lm-watermarking). We appreciate their great work!

## Cite
```
@inproceedings{ren2024robust,
  title={A Robust Semantics-based Watermark for Large Language Model against Paraphrasing},
  author={Ren, Jie and Xu, Han and Liu, Yiding and Cui, Yingqian and Wang, Shuaiqiang and Yin, Dawei and Tang, Jiliang},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2024},
  pages={613--625},
  year={2024}
}
