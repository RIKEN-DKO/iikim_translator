# iikim_translator
This repository contains translation models specifically designed for the Mayan languages, Yucatec Mayan and Chol. It's part of a project aimed at advancing neural machine translation (NMT) for indigenous languages, leveraging Spanish-trained large language models (LLMs) due to the scarcity of comprehensive parallel corpora for these languages.

## Overview
Advancing NMT for Indigenous Languages: This project focuses on a case study involving Yucatec Mayan and Chol, two Mayan languages with significant historical and cultural importance yet limited digital resources. By implementing a prompt-based approach to train both one-to-many and many-to-many models, this study demonstrates the viability of using LLMs to improve accessibility and preservation for languages with limited digital resources. The best-performing models reached a ChrF++ score of 50 on the test set, indicating a strong potential for these approaches in real-world applications.

### Project Goals
- **Develop and refine NMT systems for Mayan languages.**
- **Bridge the digital gap for Mayan language communities, enhancing access to vital services.**
- **Promote collaboration and further research in NLP for indigenous languages.**

## Training 

### M2M model
```
./train/m2m100_418M/mayan.sh
```
### T5-base-spanish model

```
./train/t5-base-spanish/mayan.sh
```

## Testing

### M2M model
```
./test/m2m100_418M/mayan.sh
```
### T5-base-spanish model

```
./test/t5-base-spanish/mayan.sh
```

## HF Models
[M2M100] (https://huggingface.co/jcrangel/iikim_translator_m2m)
[t5-base-spanish](https://huggingface.co/jcrangel/iikim_translator_t5)