# mlm-scoring-transformers

[English README](https://github.com/Ryutaro-A/mlm-scoring-transformers/blob/main/README.md)

このパッケージは[Masked Language Model Scoring(ACL2020)](https://arxiv.org/abs/1910.14659)を読んで再現実装を行ったものです．

本家様の実装ではmxnetライブラリが用いられており，日本語文に対応していません．

そこで，Hugging Face上で公開されているMasked Modelが使えるようにしたものを公開します．

全てのモデルに試してはいませんが，おそらくほとんどの事前学習モデルが利用できると思います．

## インストール
```
git clone https://github.com/Ryutaro-A/mlm-scoring-transformers.git
cd mlm-scoring-transformers
pip install .
```

### 使い方
* 日本語文のスコアを計算したいとき
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

scores = scorer.score_sentences(sample_sentences)

print('input_sentence, score')
for sentence, score in zip(sample_sentences, scores):
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

* 英語文のスコアを計算したいとき
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

## configを変更したいとき
基本はモデルの事前学習に用いられたconfigを自動で選択しますが，独自のconfigを用いることもできます．

その場合は以下のように`model_config`を設定してください．
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
model_config
```

## 連絡先
Twitter: [@ryu1104_m](https://twitter.com/ryu1104_m)'

Mail: ryu1104.as[at]gmail.com