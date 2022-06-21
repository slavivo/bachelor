# import requests
# import time  
# import msgpack
# # api-endpoint
# URL = "http://192.168.0.17/api/wifi-info"


# while(1):
#   # sending get request and saving the response as response object
#   t0 = time.time()
#   r = requests.get(url = URL)
#   t1 = time.time()
#   if((t1-t0)>1):
#     print(t1-t0)
#   # print(t0-time.time())
#   rslt = msgpack.unpackb(r.content)
#   print(len(rslt["data"]))
#   time.sleep(1)
#   # extracting data in json format
#   # data = r.json()
#   # print(data)


import requests
import json

test = {
  "id": 0,
  "deviceTypeID": 0,
  "deviceMacAdress": "string",
  "senzorData": [
    {
      "valueID": 0,
      "unitID": 0,
      "maxQueueSize": 0,
      "samplingFreq": 0,
      "values": [
        0
      ],
      "timestamps": [
        0
      ]
    }
  ]
}

hdrs = {"Content-type": "application/json",
            "Accept": "*/*"}

# print(convertedDat)
print(test)

# tests = json.dumps(test, indent = 4) 
# hdrs=json.dumps(hdrs, indent = 4)
# d="{\"id\": 0, \"deviceMacAdress\": \"f8:ca:b8:2a:e6:1c\", \"senzorData\": [{\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 0, \"unitID\": 1, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 1, \"unitID\": 1, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 2, \"unitID\": 1, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 3, \"unitID\": 2, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 4, \"unitID\": 2, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 5, \"unitID\": 2, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 6, \"unitID\": 4, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 7, \"unitID\": 4, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 8, \"unitID\": 4, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 9, \"unitID\": 3, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 10, \"unitID\": 3, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 11, \"unitID\": 3, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 12, \"unitID\": 0, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 13, \"unitID\": 0, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 14, \"unitID\": 0, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 15, \"unitID\": 0, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 16, \"unitID\": 1, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 17, \"unitID\": 1, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 18, \"unitID\": 1, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 19, \"unitID\": 1, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 20, \"unitID\": 1, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 21, \"unitID\": 1, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}]}"
d="{\"id\": 0, \"deviceMacAdress\": \"f8:ca:b8:2a:e6:1c\", \"senzorData\": [{\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 0, \"unitID\": 1, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 1, \"unitID\": 1, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 2, \"unitID\": 1, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 3, \"unitID\": 2, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 4, \"unitID\": 2, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 5, \"unitID\": 2, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 6, \"unitID\": 4, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 7, \"unitID\": 4, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 8, \"unitID\": 4, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 9, \"unitID\": 3, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 10, \"unitID\": 3, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 11, \"unitID\": 3, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 12, \"unitID\": 0, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 13, \"unitID\": 0, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 14, \"unitID\": 0, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 15, \"unitID\": 0, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 16, \"unitID\": 1, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 17, \"unitID\": 1, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 18, \"unitID\": 1, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 19, \"unitID\": 1, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 20, \"unitID\": 1, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}, {\"uid\": \"004a00223038511537353433\", \"timestamps\": [], \"valueID\": 21, \"unitID\": 1, \"deviceTypeID\": 0, \"maxQueueSize\": 3000, \"values\": []}]}"

r = requests.post("https://org-bio.azurewebsites.net/api/Device", data= d,headers=hdrs)
print(r.status_code, r.reason)
