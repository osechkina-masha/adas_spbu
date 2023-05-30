# Description
This branch contains a traffic sign detector. This detector is integrated into CVAT.

To start you need:

Ubuntu 18.04+

Docker and Docker Compose.

# How to run
- #### Clone the repo
```shell 
https://github.com/rongirl/cvat_with_traffic_sign_detector.git
```
- ####  Install nuclio 
```shell
wget https://github.com/nuclio/nuclio/releases/download/1.8.14/nuctl-1.8.14-linux-amd64
sudo chmod +x nuctl-1.8.14-linux-amd64
sudo ln -sf $(pwd)/nuctl-1.8.14-linux-amd64 /usr/local/bin/nuctl
```
- #### Build CVAT 
```shell 
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
```
- #### Create project in nuclio
```shell 
nuctl create project cvat
```
- #### Run CVAT with my detector
```shell 
sudo chmod +x detector.sh
./detector.sh
```
Open the Google Chrome browser and go to localhost:8080. My traffic sign detector is in AI tools.
