# Nepo Leon - Body Buddy

Nepo Leon loves your company!  Set him up and he'll follow you around - press his record button and he'll snap some candid videos.

## Quick Start

- Upload and install the software on your Comma Body:
  - Enable SSH
    - Power up The Comma 3X and navigate to Settings -> Network -> Advanced -> Enable SSH
    - Add your GitHub username to the device for authentication (Settings -> Network -> Advanced -> ) SSH keys
  - Upload the repo to the device:
    - Connect to your Body through SSH, clone the repo to the correct folder
```bash
$ ssh comma@$YOUR_COMMA_3X_IP_ADDRESS
$ cd /data/openpilot
$ git clone https://github.com/MikeBusuttil/openpilot.git
$ git checkout local-buddy
```
  - Install OpenCV (by first allowing write access to Python's installation directory):
```bash
$ sudo mount -o rw,remount /
$ pip install opencv-python
```
  - Re-launch openpilot
```bash
$ tmux a
# press CTRL+C twice
$ ./launch_openpilot.sh
# press ~D (the back-tick/tilde key and the D key) to get out of the tmux shell
```
- Turn it on by pressing the silver button on the wheel base to enable cruise control

## What you need

- The Comma Body
- Access to the same WiFi network as your Comma Body

## Troubleshooting

- if you loose connection to your tmux terminal, run `$ sudo systemctl restart comma`

## About

Nepo Leon was developed during the [COMMA_HACK_4 hackathon](https://blog.comma.ai/comma_hack_4/) at the Comma AI offices in sunny San Diego.  For more detailed instructions, see the [Hacakathon README](/body/README.md).

Enjoy ☺🤖
