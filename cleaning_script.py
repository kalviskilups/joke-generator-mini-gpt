#!/usr/bin/env python3

import json

if __name__ == "__main__":

    with open('reddit_jokes.json', 'r') as json_file:
        data = json.load(json_file)

    with open('jokes_parsed.txt', 'w') as parsed_jokes:
        for item in data:
            parsed_jokes.write(item['title'] + '\n')
            parsed_jokes.write(item['body'] + '\n\n')
