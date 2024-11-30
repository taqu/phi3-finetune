import os
from datasets import load_dataset
from datasets import DatasetDict
from datasets import Dataset
from openai import OpenAI
import copy

openai_key = 'XXX'
client = OpenAI(api_key=openai_key,base_url='http://172.19.128.1:9090/v1')

dataset_ids = [("LukasSonn/DoxygenStrings-Short", "DoxygenStrings-Short")]

def translate(item):
    template = """You are a highly skilled professional Japanese-English and English-Japanese translator. Translate the given text accurately, taking into account the context and specific instructions provided. Steps may include hints enclosed in square brackets [] with the key and value separated by a colon:. Only when the subject is specified in the Japanese sentence, the subject will be added when translating into English. If no additional instructions or context are provided, use your expertise to consider what the most appropriate context is and provide a natural translation that aligns with that context. When translating, strive to faithfully reflect the meaning and tone of the original text, pay attention to cultural nuances and differences in language usage, and ensure that the translation is grammatically correct and easy to read. After completing the translation, review it once more to check for errors or unnatural expressions. For technical terms and proper nouns, either leave them in the original language or use appropriate translations as necessary. Take a deep breath, calm down, and start translating.

<start_of_turn>### Instruction:
Translate English to Japanese.

### Input:
{0}

<end_of_turn>
<start_of_turn>### Response:
"""
    query = template.format(item)
    completion = client.chat.completions.create(
        model='c3tr',
        messages=[
            {"role": "user", "content": query}
        ]
    )
    answer = completion.choices[0].message.content
    start = answer.find('/**')
    if start<0:
        return None
    end = answer.find('*/')
    if end<0:
        return None
    end += 2
    item['question'] = 'Create a doxygen comment for the following C++ Function in Japanese.'
    item['answer'] = answer[start:end]
    return item

def translate_test(item):
    template = """You are a highly skilled professional Japanese-English and English-Japanese translator. Translate the given text accurately, taking into account the context and specific instructions provided. Steps may include hints enclosed in square brackets [] with the key and value separated by a colon:. Only when the subject is specified in the Japanese sentence, the subject will be added when translating into English. If no additional instructions or context are provided, use your expertise to consider what the most appropriate context is and provide a natural translation that aligns with that context. When translating, strive to faithfully reflect the meaning and tone of the original text, pay attention to cultural nuances and differences in language usage, and ensure that the translation is grammatically correct and easy to read. After completing the translation, review it once more to check for errors or unnatural expressions. For technical terms and proper nouns, either leave them in the original language or use appropriate translations as necessary. Take a deep breath, calm down, and start translating.

<start_of_turn>### Instruction:
Translate English to Japanese.

### Input:
{0}

<end_of_turn>
<start_of_turn>### Response:
"""

    query = template.format(item)
    print("query\n"+query)
    completion = client.chat.completions.create(
        model='c3tr',
        messages=[
            {"role": "user", "content": query}
        ]
    )
    answer = completion.choices[0].message.content
    return answer

count = 0
for dataset_id in dataset_ids:
    dataset_dict = load_dataset(dataset_id[0])
    print(dataset_dict)
    new_dataset_dict = DatasetDict()
    for key in dataset_dict.keys():
        dataset = dataset_dict[key]
        new_dataset = []
        for item in dataset:
            answer = item['answer']
            if not answer:
                continue
            start = answer.find('/**')
            if start<0:
                continue
            end = answer.find('*/')
            if end<0:
                continue
            end += 2
            result = answer[start:end]
            print(result)
            item['answer'] = result
            new_dataset.append(item)
            item = translate(copy.deepcopy(item))
            if item is None:
                pass
            else:
                print(item['answer'])
                new_dataset.append(item)

        new_dataset = Dataset.from_list(new_dataset);
        new_dataset_dict[key] = new_dataset

    print(new_dataset_dict)
    new_dataset_dict.save_to_disk(dataset_id[1])

