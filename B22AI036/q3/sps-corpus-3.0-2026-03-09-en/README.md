# *English* &mdash; English (`en`)

This datasheet is for sps-corpus-3.0-2026-03-09 of the Mozilla Common Voice *Spontaneous Speech* dataset for English [English - `en`]. The dataset contains 6382 clips representing 18.38 hours of recorded speech (7.09 hours validated) from 596 speakers.

## Language

English is a West Germanic language with origins in England. There are an estimated 1.5 billion English speakers, making it the most widely spoken language in the world. English is commonly learned as a second language in many countries.

## Data splits for modelling

The dataset clips are categorised by transcription status and training-set assignment. The following tables summarise the distribution.

### Audio clips

| Bucket | Clips | % |
| --- | --- | --- |
| Transcribed & Validated | 1,897 | 29.7% |
| Transcribed & Pending | 84 | 1.3% |
| Not transcribed | 4,401 | 69.0% |

### Training splits

| Bucket | Clips | % |
| --- | --- | --- |
| Train | 973 | 15.2% |
| Dev | 351 | 5.5% |
| Test | 573 | 9.0% |
| Unassigned | 4,485 | 70.3% |

Training split coverage: 1,897 of 1,897 transcribed & validated clips (100.0%)

## Transcriptions

### Transcription status

| Bucket | Clips | % |
| --- | --- | --- |
| Validated | 1,897 | 95.8% |
| Pending | 84 | 4.2% |
| Edited | 203 | 10.3% |

### Writing system

The English writing system is based off of the latin alphabet.

#### Symbol table

`a b c d e f g h i j k l m n o p q r s t u v w x y z`

### Samples

#### Questions

There follows a randomly selected sample of questions used in the corpus.

1. *How you treat your neighbours?*
2. *How do you think a person can keep from becoming a workaholic?*
3. *What language cues accompany hopeful statements (tone, gestures, interjections)?*
4. *What are the qualities of a good human being?*
5. *Describe a visit to an ocean, lake or river.*

#### Responses

There follows a randomly selected sample of transcribed responses from the corpus.

1. *[silence]*
2. *I eat cake*
3. *I adore cake. It's probably my greatest weakness in food.   umm favorite cake would be fruit cake and if you want to add icing and marzipan to that I'd welcome that. That thats going to be my absolutely top fruit cake is gonna be celebration cake with marzipan and icing.   Uh the next choice would be flapjack. I'll tell you that again flapjack.  There's something really nice about flapjack. its honest. It's oats, its very good for you, um although probably the honey with which its held together isn't quite so good for you.  but yes those are the top two. Umm I also like custard tarts. Don't know why, just always have liked custard tarts.   there we go.*
4. *Where do you buy clothes?*
5. *What was your first date like?*

### Fields

Each row of a `tsv` file represents a single audio clip, and contains the following information:

- `client_id` - hashed UUID of a given user
- `audio_id` - numeric id for audio file
- `audio_file` - audio file name
- `duration_ms` - duration of audio in milliseconds
- `prompt_id` - numeric id for prompt
- `prompt` - question for user
- `transcription` - transcription of the audio response
- `votes` - number of people that who approved a given transcript
- `age` - age of the speaker[^1]
- `gender` - gender of the speaker[^1]
- `language` - language name
- `split` - for data modelling, which subset of the data does this clip pertain to
- `char_per_sec` - how many characters of transcription per second of audio
- `quality_tags` - some automated assessment of the transcription--audio pair, separated by `|`
  - `transcription-length` - character per second under 3 characters per second
  - `speech-rate` - characters per second over 30 characters per second
  - `short-audio` - audio length under 2 seconds
  - `long-audio` - audio length over 5 minutes

---

[^1]: For a full list of age, gender, and accent options, see the [demographics spec](https://github.com/common-voice/common-voice/blob/main/web/src/stores/demographics.ts). These will only be reported if the speaker opted in to provide that information.

## Get involved

### Community links

- [Common Voice translators on Pontoon](https://pontoon.mozilla.org/en/common-voice/contributors/)
- [Common Voice Communities](https://github.com/common-voice/common-voice/blob/main/docs/COMMUNITIES.md)

### Discussions

- [Common Voice on Matrix](https://chat.mozilla.org/#/room/#common-voice:mozilla.org)
- [Common Voice on Discourse](https://discourse.mozilla.org/t/about-common-voice-readme-first/17218)
- [Common Voice on Discord](https://discord.gg/9QTj9zwn)
- [Common Voice on Telegram](https://t.me/mozilla_common_voice)

### Contribute

- [Contribute questions](https://commonvoice.mozilla.org/spontaneous-speech/beta/question)
- [Validate questions](https://commonvoice.mozilla.org/spontaneous-speech/beta/validate)
- [Answer questions](https://commonvoice.mozilla.org/spontaneous-speech/beta/prompts)
- [Transcribe recordings](https://commonvoice.mozilla.org/spontaneous-speech/beta/transcribe)
- [Validate transcriptions](https://commonvoice.mozilla.org/spontaneous-speech/beta/check-transcript)

## Licence

This dataset is released under the [Creative Commons Zero (CC-0)](https://creativecommons.org/public-domain/cc0/) licence. By downloading this data you agree to not determine the identity of speakers in the dataset.
