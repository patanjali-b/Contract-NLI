# Contract-NLI Implementation Guide

In this project, we have implemented three variants of BERT for Natural Language Inference (NLI) classification. Additionally, we have introduced a novel approach for Evidence Inferencing. The code is written in Python 3.11.5 and PyTorch 2.1.0.

## Part 1: DistilBERT

To train and save the DistilBERT model, follow these steps:

```bash
cd DistilBERT
python3 DistilBERT.py
```

## Part 2: MobileBERT

To train and save the MobileBERT model, follow these steps:

```bash
cd MobileBERT
python3 MobileBERT.py
```

## Part 3: AlBERT

To train and save the AlBERT model, follow these steps:

```bash
cd AlBERT
python3 AlBERT.py
```

## Part 4: Evidence Inferencing

To train and save the Evidence Inferencing model, follow these steps:

```bash
cd BertEI
python3 EntailmentBERT.py
python3 ContradictionBERT.py
```

To test the above save models, follow these steps:

```bash
cd BertEI
python3 TestingBERT.py
```



## Final Model

For evidence inferencing, we have utilized the pretrained models of DistilBERT, MobileBERT, and ALBERT. Follow these steps:

1. **Change the directory to the home folder.**
   
2. **Update the path of the pretrained model in the `model.py` file.**

3. **Run the following command:**

   ```bash
   python3 model.py
    ```

#### [The link to the models is here, click me](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/patanjali_b_research_iiit_ac_in/EnBKyavrLtlHiUJ8WJmFyusB32oHBwlRrg34c5jrIojfGQ?e=qeFfw2)

#### [The link to the slides is here, click me](https://www.canva.com/design/DAFzTnMjhR0/DMXuGuzsDJhfkawyqc3_NA/edit?utm_content=DAFzTnMjhR0&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)




