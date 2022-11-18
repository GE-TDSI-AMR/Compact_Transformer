import wget
import os

data_sets_urls = {
    'cifar10':
        'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
    'cifar100':
        'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
}

data_sets_path = 'datasets/'


def data_download():
    try:
        os.mkdir(data_sets_path)
    except OSError as error:
        print("/datasets folder already exist")

    for data_set, url in data_sets_urls.items():
        path = f'{data_sets_path}{data_set}.tar.gz'
        if not os.path.exists(path):
            wget.download(url, path)


if __name__ == "__main__":
    data_download()
    print("abdoul")
