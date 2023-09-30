#!/usr/bin/env python3
import time, json
from cereal import messaging

def main():
  pm = messaging.PubMaster(['customReservedRawData1'])
  for i in range(2):
    time.sleep(2)
    msg = messaging.new_message()
    msg.customReservedRawData1 = json.dumps({"hi": "bro", "whats": "up"}).encode()
    pm.send('customReservedRawData1', msg)
  

if __name__ == "__main__":
  main()
