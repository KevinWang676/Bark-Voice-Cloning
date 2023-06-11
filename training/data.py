import random
import requests
import os, glob

# english literature
books = [
     'https://www.gutenberg.org/cache/epub/1513/pg1513.txt',
     'https://www.gutenberg.org/files/2701/2701-0.txt',
     'https://www.gutenberg.org/cache/epub/84/pg84.txt',
     'https://www.gutenberg.org/cache/epub/2641/pg2641.txt',
     'https://www.gutenberg.org/cache/epub/1342/pg1342.txt',
     'https://www.gutenberg.org/cache/epub/100/pg100.txt'
 ]

#default english
# allowed_chars = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_+=\"\':;[]{}/<>,.`~\n\\'

#german
allowed_chars = ' aäbcdefghijklmnoöpqrsßtuüvwxyzABCDEFGHIJKLMNOÖPQRSTUÜVWXYZ0123456789!@#$%^&*()-_+=\"\':;[]{}/<>,.`~\n\\'


def download_book(book):
    return requests.get(book).content.decode('utf-8')


def filter_data(data):
    print('Filtering data')
    return ''.join([char for char in data if char in allowed_chars])


def load_books(fromfolder=False):
    text_data = []
    if fromfolder:
        current_working_directory = os.getcwd()
        print(current_working_directory)
        path = 'text'
        for filename in glob.glob(os.path.join(path, '*.txt')):
            with open(os.path.join(os.getcwd(), filename), 'r') as f: # open in readonly mode
                print(f'Loading {filename}')
                text_data.append(filter_data(str(f.read())))
    else:
        print(f'Loading {len(books)} books into ram')
        for book in books:
            text_data.append(filter_data(str(download_book(book))))
    print('Loaded books')
    return ' '.join(text_data)


def random_split_chunk(data, size=14):
    data = data.split(' ')
    index = random.randrange(0, len(data))
    return ' '.join(data[index:index+size])
