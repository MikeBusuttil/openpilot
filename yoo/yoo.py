#!/usr/bin/env python3
import time, json
from cereal import messaging

def main():
  pm = messaging.PubMaster(['customReservedRawData1'])
  while True:
    for i in range(250):
      msg = messaging.new_message()
      msg.customReservedRawData1 = json.dumps({"back": 0, "left": 1}).encode()
      pm.send('customReservedRawData1', msg)
    time.sleep(5)
  

if __name__ == "__main__":
  main()
