import mlmt

pretrained_model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'

my_scorer = mlmt.MLMScorer(pretrained_model_name, use_cuda=False)

jp = [
    'お母さんが行けるなら、わたしは行くのをやめるよ。うちから二人も出ることはないから。',
    'お母さんが行けると、わたしは行くのをやめるよ。うちから二人も出ることはないから。',
    'お母さんが行けたなら、わたしは行くのをやめるよ。うちから二人も出ることはないから。',
    'お母さんが行けるのだったら、わたしは行くのをやめるよ。うちから二人も出ることはないから。',
    '日本酒を飲めば、駅の反対側にある「××酒蔵」が一番だね。とにかく品揃えが抜群だよ。',
    '日本酒を飲むなら、駅の反対側にある「××酒蔵」が一番だね。とにかく品揃えが抜群だよ。',
    '日本酒を飲むんだったら、駅の反対側にある「××酒蔵」が一番だね。とにかく品揃えが抜群だよ。',
    '日本酒を飲むと、駅の反対側にある「××酒蔵」が一番だね。とにかく品揃えが抜群だよ。',
]

scores = my_scorer.score_sentences(
    sentences=jp,
    get_token_likelihood=True
)

print('input_sentence, score')
for sentence, score in zip(jp, scores):
    print(sentence, score["all"])
    print(score["token"])


my_scorer = mlmt.MLMScorer('bert-base-uncased', use_cuda=False)

en = [
    'Due to the rain, our performance in the game was far from perfect.',
    'Due to the rain, our performance in the game was apart from perfect.',
    'Due to the rain, our performance in the game was different from perfect.',
    'Due to the rain, our performance in the game was free from perfect.',
]

scores = my_scorer.score_sentences(
    sentences=en,
    get_token_likelihood=True
)

print('input_sentence, score')
for sentence, score in zip(en, scores):
    print(sentence, score["all"])
    print(score["token"])