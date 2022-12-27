# Getting started locally

Please checkout the python notebook in this repo for some more info and help. This is just a rough setup to get you familiar with everything.

## Testing the app on K8s locally

### Installing a k8s cluster

<details>
  <summary>➡️  Install a k3s cluster with `smol-k8s-lab`</summary>

#### Using smol-k8s-lab
I wrote this tool this year for working locally on k8s projects.
You can check it out on [github](https://github.com/small-hack/smol-k8s-lab).

It installs metallb, the nginx ingress controller, and can also install argocd,
as well as the external secrets provider.

```bash
pip3.11 install smol-k8s-lab

# make sure you create this config directory (will be automated soon)
mkdir -p ~/.config/smol-k8s-lab

# set this to a free IP on your network (don't forget the CIDR notation!) and then you can use it in your local DNS
# you can setup an A record for your domain in your pihole 'local DNS' if you're using that, or ping me, and I can help you with your local router!
echo -e "metallb_address_pool:\n  - 192.168.42.42/32" > ~/.config/smol-k8s-lab/config.yaml

# this is used for SSL with lets encrypt
echo "email: name@email.com" >> ~/.config/smol-k8s-lab/config.yaml

# this is the log level, which I set to debug so you can see everything going on
echo -e "log:\n  level: debug" >> ~/.config/smol-k8s-lab/config.yaml

# k3s is best on Linux (note: torch is not made for macOS and will not run on a mac with no GPU)
# NOTE: THIS REQUIRES SUDO ACCESS
smol-k8s-lab k3s
```

</details>

### Installing the manifests

After you've installed your k8s cluster with one of the methods above,
it's time to get the manifests for the interview assignment installed.

```bash
kubectl apply -f k8s_manifests/
```

### Testing the k8s `infer_image` endpoint

```bash
# do some port forwarding to test locally and avoid networking headaches
# this will run in the foreground
kubectl port-forward deployment/infer-sat-image-flask-app 5000:8080
```

If you have a satellite tile locally, I created a small cli script to crop it in `crop_satellite_img.py` for convienence.

In another terminal, try the following where `cropped_img.tif` is replaced by the path to your 512x512 sat image crop:
```bash
# the /0 is a boolean for gzip (meaning do not gzip), I didn't have time to implement the gzip enabled
curl -F '@file=cropped_img.tif' 127.0.0.1:5000/infer_image/0 -o test.pkl
```

To test your load your numpy array from the pickle file you can do:
```python
import numpy as np
np.load('test.pkl', allow_pickle=True)
```
:tada: and that's it!

## Building the Dockerfile and testing locally

### Building

```bash
# 0.0.1 can be any version, but remember to tick it up when testing a new build
docker build . -t jessebot/infer-sat-img-api:0.0.1
```

### Running locally

```bash
# forward port 8080 in the docker image to your host port 5000
# mount the container /tmp directory as a volume to your local /tmp directory
docker run -it -p 5000:8080 -v /tmp:/tmp jessebot/infer-sat-img-api:0.0.1
```

## testing `infer_image`

```bash
curl -F '@file=/home/USER/file.tiff' 127.0.0.1:5000/infer_image/0 -o test.pkl
```

## Testing the endpoint
To test your load your numpy array from the pickle file you can do:
```python
import numpy as np
np.load('test.pkl', allow_pickle=True)
```
