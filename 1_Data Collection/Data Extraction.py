import requests
from os.path import join
import multiprocessing as mp
import json
import pandas as pd
import os

csv_address_file = '1_Data Collection/contracts.csv'
os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')
keys = 'I49P5NV5JKT9QP9HNKZUSWTQQCQGW5HVVX'
# Get API Keys from https://etherscan.io
present_dir = os.getcwd()


def parsesc_json(filename):
    with open(filename) as access_json:
        read_content = json.load(access_json)
    results = read_content['result']
    file_to_create = filename.replace('json','sol')
    for result in results:
        with open(file_to_create, 'w') as file:
            file.write(result['SourceCode'])

def parseop_json(filename):
    with open(filename) as access_json:
        read_content = json.load(access_json)
    result = read_content['result']
    file_to_create = filename.replace('json','sol')
    with open(file_to_create, 'w') as file:
        file.write(result)
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
        self.totalSourceFiles = 0
        self.totalOpFiles = 0
        self.sourceCompleted = 0
        self.opCompleted = 0

    def incCount(self):
        self.totalSourceFiles += 1
        self.totalOpFiles += 1

    def getSourceTotal(self):
        return self.totalFiles

    def getOpTotal(self):
        return self.totalFiles

    def sourceCompletedCallback(self, res=''):
        self.sourceCompleted += 1
        print(self.opCompleted, " files Completed out of ", self.totalSourceFiles)

    def opCompletedCallback(self, res=''):
        self.opCompleted += 1
        print(self.opCompleted, " files Completed out of ", self.totalOpFiles)

def extractFromEthereum():
    pool = mp.Pool(2)  # X keys, calls per second = 5, roughly X*3=Y works here
    df = pd.read_csv(csv_address_file)
    hashes = df["address"].tolist()
    ProcessingResults = []
    countObj = CheckCount()
    for i in range(len(hashes)):
        contPath = join("1_Data Collection/Source_code/json/", hashes[i] + ".json")
        ProcessingResults = pool.apply_async(etherDownloadApi, args=(
        contPath, 'getsourcecode', 'contract', hashes[i], keys),
                                             callback=countObj.sourceCompletedCallback)
        contPathOp = join("1_Data Collection/Operation_code/json/", hashes[i] + ".json")
        ProcessingResultsOp = pool.apply_async(etherDownloadApi, args=(
            contPathOp, 'getopcode', 'opcode', hashes[i], keys),
                                             callback=countObj.opCompletedCallback)
        countObj.incCount()

    pool.close()
    pool.join()


if __name__ == '__main__':
    df = pd.read_csv(csv_address_file)
    hashes = df["address"].tolist()

    extractFromEthereum()

    try:
        for i in range(len(hashes)):
            parsesc_json("1_Data Collection/Source_code/json/"+hashes[i] + ".json")
            print(hashes[i] + " source code completed")
            parseop_json("1_Data Collection/Operation_code/json/" + hashes[i] + ".json")
            print(hashes[i] + " operation code completed")
    except:
        print(hashes[i]+" cannot")
