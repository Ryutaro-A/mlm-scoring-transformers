# mlm-scoring-transformers

[日本語 README](https://github.com/Ryutaro-A/mlm-scoring-transformers/blob/main/README_JA.md)

This package is a reproduced implementation of [Masked Language Model Scoring (ACL2020)](https://arxiv.org/abs/1910.14659).

The original implementation uses the mxnet library, which does not support Japanese.

Therefore, we are releasing a version that can be used with the Masked Model published on Hugging Face.

We have not tried it on all models, but we believe that most of the pre-trained models can be used.


## Installation
```
git clone https://github.com/Ryutaro-A/mlm-scoring-transformers.git
cd mlm-scoring-transformers
pip install .
```

### Get Started
* To calculate scores for Japanese sentences.
```python
import mlmt

pretrained_model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'

scorer = mlmt.MLMScorer(pretrained_model_name, use_cuda=False)

japanese_sample_sentences = [
    'お母さんが行けるなら、わたしは行くのをやめるよ。うちから二人も出ることはないから。',
    'お母さんが行けると、わたしは行くのをやめるよ。うちから二人も出ることはないから。',
    'お母さんが行けたなら、わたしは行くのをやめるよ。うちから二人も出ることはないから。',
    'お母さんが行けるのだったら、わたしは行くのをやめるよ。うちから二人も出ることはないから。',
    '日本酒を飲めば、駅の反対側にある「××酒蔵」が一番だね。とにかく品揃えが抜群だよ。',
    '日本酒を飲むなら、駅の反対側にある「××酒蔵」が一番だね。とにかく品揃えが抜群だよ。',
    '日本酒を飲むんだったら、駅の反対側にある「××酒蔵」が一番だね。とにかく品揃えが抜群だよ。',
    '日本酒を飲むと、駅の反対側にある「××酒蔵」が一番だね。とにかく品揃えが抜群だよ。',
]

scores = scorer.score_sentences(japanese_sample_sentences)

print('input_sentence, score')
for sentence, score in zip(japanese_sample_sentences, scores):
    print(sentence, score)

# >> input_sentence, score
# お母さんが行けるなら、わたしは行くのをやめるよ。うちから二人も出ることはないから。 -72.90809887713657
# お母さんが行けると、わたしは行くのをやめるよ。うちから二人も出ることはないから。 -75.87569694537336
# お母さんが行けたなら、わたしは行くのをやめるよ。うちから二人も出ることはないから。 -65.31722020490005
# お母さんが行けるのだったら、わたしは行くのをやめるよ。うちから二人も出ることはないから。 -86.46473170552028
# 日本酒を飲めば、駅の反対側にある「××酒蔵」が一番だね。とにかく品揃えが抜群だよ。 -85.50868926288888
# 日本酒を飲むなら、駅の反対側にある「××酒蔵」が一番だね。とにかく品揃えが抜群だよ。 -81.26314979794296
# 日本酒を飲むんだったら、駅の反対側にある「××酒蔵」が一番だね。とにかく品揃えが抜群だよ。 -82.7387441759266
# 日本酒を飲むと、駅の反対側にある「××酒蔵」が一番だね。とにかく品揃えが抜群だよ。 -92.14111483963103
```

* To calculate scores for English sentences.
```python
import mlmt

pretrained_model_name = 'bert-base-uncased'

scorer = mlmt.MLMScorer(pretrained_model_name, use_cuda=False)

english_sample_sentences = [
    'Due to the rain, our performance in the game was far from perfect.',
    'Due to the rain, our performance in the game was apart from perfect.',
    'Due to the rain, our performance in the game was different from perfect.',
    'Due to the rain, our performance in the game was free from perfect.',
]

scores = scorer.score_sentences(english_sample_sentences)

print('input_sentence, score')
for sentence, score in zip(english_sample_sentences, scores):
    print(sentence, score)

# >> input_sentence, score
# Due to the rain, our performance in the game was far from perfect. -13.874692459549525
# Due to the rain, our performance in the game was apart from perfect. -15.486674794020251
# Due to the rain, our performance in the game was different from perfect. -16.62563831794064
# Due to the rain, our performance in the game was free from perfect. -20.5683701854279
```

## To change model config
Basically, the config used to pre-train the model is automatically selected, but you can also use your own config.

In that case, set `model_config` as follows.
```python
config = transformers.BertConfig(
        hidden_act="gelu",
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
        model_type="bert",
        num_attention_heads=16,
        num_hidden_layers=24,
        pad_token_id=0,
        tokenizer_class="BertJapaneseTokenizer",
        type_vocab_size=2,
        vocab_size=32768,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.37
)
scorer = mlmt.MLMScorer(pretrained_model_name, model_config=config, use_cuda=False)
```

## License
This software is released under the MIT License, see LICENSE.txt.

## Contacts
Twitter: [@ryu1104_m](https://twitter.com/ryu1104_m)

Mail: ryu1104.as[at]gmail.com