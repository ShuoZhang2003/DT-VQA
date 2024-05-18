# DT-VQA: Exploring the Capabilities of Large Multimodal Models on Dense Text

<div align="center">

Shuo Zhang, Biao Yang, Zhang Li, Zhiyin Ma, Yuliang Liu†, Xiang Bai

</div>

<div align="center">

<strong>Huazhong University of Science and Technology</strong>

</div>

<p align="center">

<a href="https://arxiv.org/abs/2405.06706">Paper</a> | <a href="https://huggingface.co/shuozhang2/DT-VQA">DT-VQA dataset</a>

<!-- | &nbsp&nbsp<a href="Monkey Model">Monkey Models</a>&nbsp ｜ &nbsp <a href="updating">Tutorial</a> -->

</p>

-----

While large multi-modal models (LMM) have shown notable progress in multi-modal tasks, their capabilities in tasks involving dense textual content remains to be fully explored. Dense text, which carries important information, is often found in documents, tables, and product descriptions. Understanding dense text enables us to obtain more accurate information, assisting in making better decisions. To further explore the capabilities of LMM in complex text tasks, we propose the DT-VQA dataset, with 170k question-answer pairs. In this paper, we conduct a comprehensive evaluation of GPT4V, Gemini, and various open-source LMMs on our dataset, revealing their strengths and weaknesses. Furthermore, we evaluate the effectiveness of two strategies for LMM: prompt engineering and downstream fine-tuning. We find that even with automatically labeled training datasets, significant improvements in model performance can be achieved. We hope that this research will promote the study of LMM in dense text tasks.


## Dataset

We have open-sourced the data. You can download it at [DT-VQA](https://huggingface.co/shuozhang2/DT-VQA).

Note: The training set is organized in the Monkey multi-round dialogue training data format.

## Evaluate

The test code for evaluating models in the paper can be found in [scripts](./scripts). Before conducting the evaluation, you need to configure the model weights and environment based on the official code link provided in the scripts. If you want to evaluate other models, please edit the "TODO" things in [example](./example.py).

## Citing

If you wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX
@misc{zhang2024exploring,
      title={Exploring the Capabilities of Large Multimodal Models on Dense Text}, 
      author={Shuo Zhang and Biao Yang and Zhang Li and Zhiyin Ma and Yuliang Liu and Xiang Bai},
      year={2024},
      eprint={2405.06706},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```