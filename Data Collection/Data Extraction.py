import requests
from os.path import join
import multiprocessing as mp
import json
import pandas as pd
import os

csv_address_file = 'data/contracts.csv'
report_dir = 'home'
keys = 'I49P5NV5JKT9QP9HNKZUSWTQQCQGW5HVVX'
# Get API Keys from https://etherscan.io
present_dir = os.getcwd()


def parse_json(filename):
    print(filename)
    with open(filename) as access_json:
        read_content = json.load(access_json)
    results = read_content['result']
    file_to_create = "source_code/" + filename.replace('.json', '.sol')
    file_to_create = file_to_create.replace('/home', '')
    for result in results:
        with open(file_to_create, 'w') as file:
            file.write(result['SourceCode'])


def etherDownloadApi(file_path, action, module, add, key):
    url = 'https://api.etherscan.io/api?module=' + module + '&action=' + action + '&address=' + add + '&apikey=' + key
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        with open(file_path, 'w') as f:
            json.dump(data, f)
            print(add+" json completed!")


class CheckCount:
    def __init__(self):
        self.totalFiles = 0
        self.completed = 0

    def incCount(self):
        self.totalFiles += 1

    def getTotal(self):
        return self.totalFiles

    def completedCallback(self, res=''):
        self.completed += 1
        print(self.completed, " files Completed out of ", self.totalFiles)


def extractFromEthereum():
    pool = mp.Pool(1)  # X keys, calls per second = 5, roughly X*3=Y works here
    df = pd.read_csv(csv_address_file)
    hashes = df["address"].tolist()
    ProcessingResults = []
    countObj = CheckCount()
    for i in range(len(hashes)):
        contPath = join(report_dir, hashes[i] + "_ext.json")
        ProcessingResults = pool.apply_async(etherDownloadApi, args=(
        contPath, 'getsourcecode', 'contract', hashes[i], keys),
                                             callback=countObj.completedCallback)
        countObj.incCount()

    pool.close()
    pool.join()


if __name__ == '__main__':
    df = pd.read_csv(csv_address_file)
    hashes = df["address"].tolist()

    extractFromEthereum()

    try:
        for i in range(len(hashes)):
            parse_json("home/"+hashes[i] + "_ext.json")
            print(hashes[i]+" source code completed")
    except:
        print(hashes[i]+" cannot")
