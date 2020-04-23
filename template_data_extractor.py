"""
This python file extracts data from json file which contains template data.

@author: Mitul Prajapati (mituldprajapati@gmail.com)
"""
import pandas as pd
import json
from collections import defaultdict
import re
import os


def main():
    data_dir = 'data'
    stopwords = open(os.path.join(data_dir, 'stopwords_en.txt'), 'r').read().split('\n')

    with open(os.path.join(data_dir, 'sample_pl_data.json'), 'r') as file:
        dataset = json.load(file)

    print('[INFO]: File Loaded')

    # convert template list to dict
    template_data = defaultdict()
    for i in range(len(dataset)):
        template_data[i] = dataset[i]

    def remove_stopwords(data):
        # split by punctuation and remove all symbols except characters and numbers
        words = re.split('[/.^#@! \']', re.sub('[^A-Za-z0-9 ]+', '', data))

        processed_words = []
        # remove stop words
        for w in range(len(words)):
            if words[w] not in stopwords:
                processed_words.append(words[w])
        return processed_words

    print('[INFO]: Extracting Data ...')

    # generate data
    template_id_list = []
    template_name_list = []
    keywords_list = []
    sub_industry_list = []
    industry_list = []
    use_count_list = []
    rating_list = []

    # for items
    for i in range(len(template_data)):
        template_id_list.append(template_data[i]['template_id'])
        template_name_list.append(template_data[i]['template_data']['metadata']['name'])
        sub_industry_list.append(template_data[i]['template_data']['metadata']['subindustry'])
        industry_list.append(template_data[i]['template_data']['metadata']['industry'])
        use_count_list.append(template_data[i]['template_data']['metrics']['use_count'])
        rating_list.append(template_data[i]['template_data']['metrics']['rating'])
        words_list = []
        items_list = template_data[i]['items']
        for j in range(len(items_list)):
            if items_list[j]['type'] == 'question':
                if 'label' in items_list[j].keys():
                    words_list.extend(remove_stopwords(items_list[j]['label']))
        keywords_list.append(words_list)

    templates_df = pd.DataFrame({'template_id': template_id_list, 'template_name': template_name_list,
                                 'sub_industry': sub_industry_list, 'industry': industry_list,
                                 'use_count': use_count_list, 'rating': rating_list, 'keywords': keywords_list})
    templates_df.to_csv(os.path.join(data_dir, 'templates_data.csv'), sep='|')
    print('[INFO]: Data saved successfully at {0}'.format(os.path.join(data_dir, 'templates_data.csv')))


if __name__ == '__main__':
    main()
