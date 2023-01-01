# Assignment - Part 1
You'll find [GCP-arch-options.drawio](./GCP-arch-options.drawio) as well as [a png](./GCP-arch-options.png) in this directory as the diagram requested for the first part of the assignment. My expertise is mostly in on-prem cloud and AWS, but I've also worked with GCP a bit in past roles, so I did my best on this one, but the time limit of 30 minutes didn't leave too much time for research to optimize the services we're using. Instead, I've created some basic examples explaining different ways to optimize generally, after doing some basic research of AWS -> GKE equivilents. The diagram will cover:

- GKE directly
- Hybrid cloud setup with GKE and a local k8s distro
- Serverless on GKE (though I would need some more time to actually flesh these out)

Assumption was made that this would be to deploy something similar to the assignment. Please also assume that we would have a mirrored enviornment for both prod and staging.

<hr>

# Assignment - Part 2

I spent more time on this, because there were some bugs in the python notebook and I had to get up to speed on pytorch and testing k8s on *GPU optimized* local metal. It was worth it. This was super fun :D

Note: An earlier version of this repo did not include the GPU optimized Dockerfile, but it should be there now, alone with links on how to install the Nvidia drivers for your host  machine.

### Tools used

- tested on a machine running Ubuntu 22.04 LTS with 4 cores and an Nvidia RTX2060
  - Note: I've included both an unoptimized and GPU-optimized Docker images for this assignement, this way you can still test if you have no available GPU.
- libraries in `requirements.txt`:
  - argparse (take options in script)
  - flask (serve endpoint)
  - the Overstory pre-provided `utils.py` as well as the libraries it depended on (e.g. rasterio, pytorch, matplotlib...)
- [docker] as well as nvidia's docker.
- [smol-k8s-lab] to quickly get a [k3s] cluster running
- [k3s] was used as a slim distro that could still scale out (instead of using something like [KinD] or [minikube])
- [k9s] was used for monitoring the cluster
- [euporie] for viewing and writing notebooks (this worked surprisingly well, but was a bit slow to render the graphics)
- [iTerm2] and [wezterm] - both making use of [sixel] and their own image libs for graphics/charts in the terminal
- [zellij] as a terminal multiplexer for managing sessions, tabs, panes, and floating windows in the terminal (especially via SSH)
- [vim] (text editor) - using the [YouCompleteMe] for LSP, [Semshi] for semantic highlighting, and [ruff] for linting
- [w3m] (simple browser with [sixel], mostly used for brushing up on docs)

### Before you start

If you have a satellite tile locally, I created a small cli script to crop it in `crop_satellite_img.py` for convienence. The script defaults to the same tile that is downloaded in the assignment notebook, but you can also pass in any other tif like so:

```bash
python3.11 crop_setellite_image.py -s /path/to/your/sat_tile.tif
```

# Getting started locally with K8s

Please checkout the python notebook in this repo for some more info and help. This is just a rough setup to get you familiar with everything.

## Installing a k8s cluster

### Using [smol-k8s-lab]
I wrote this tool earlier this year for working locally on k8s projects.

It installs metallb, the nginx ingress controller, and can optionally install [Argo CD],
as well as the [External Secrets Operator]. For this assignment though,
we're mostly interested in the bare bones that smol-k8s-lab can setup for you,
so we'll save the other features for another time.

```bash
# this should install everything you need, but checkout smol-k8s-lab --help for extra tooling
pip3.11 install smol-k8s-lab

# make sure you create this config directory (will be automated soon)
mkdir -p ~/.config/smol-k8s-lab

# set this to a free IP on your network (don't forget the CIDR notation!) and then you can use it in your local DNS
# you can setup an A record for your domain in your pihole 'local DNS' if you're using that, or ping me, and I can help you with your local router!
echo -e "metallb_address_pool:\n  - 192.168.42.42/32" > ~/.config/smol-k8s-lab/config.yaml

# this is used for SSL with Lets Encrypt - change to your email if you'd like to play with SSL
echo "email: name@email.com" >> ~/.config/smol-k8s-lab/config.yaml

# this is the log level, which I set to debug so you can see everything going on
echo -e "log:\n  level: debug" >> ~/.config/smol-k8s-lab/config.yaml

# k3s is best on Linux (note: pytorch is not made for macOS and will not run on a mac with no GPU)
# NOTE: THIS REQUIRES SUDO ACCESS
smol-k8s-lab k3s
```

#### Note on Networking
`smol-k8s-lab` will setup your endpoint to run at the first IP available in your
provided IP range. So, in the above example this would be 192.168.42.42.
Below, we'll be installing a k8s ingress that uses the hostname: interview.overstory-test.com

This means that after you install the manifests below, you can go into your local router settings and create a local DNS A record entry to have 192.168.42.42 point to interview.overstory-test.com.

Then, if that entry works, you can skip the sections here and in the notebook about port forwarding locally, and replace `127.0.0.1:5000` with `interview.overstory-test.com` in the `curl` examples.

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

The docker images are huge, but available here:

- [Docker image for machine with no GPU](https://hub.docker.com/r/jessebot/infer-sat-image-api).
- [Docker image for machine with nvidia GPU](https://hub.docker.com/r/jessebot/).

Unfortunately, I don't have the hardware to test GPUs that aren't nvidia at this time.

If you want to use the GPU optimized image above, you need to first install the [nvidia drivers] on your local machine. Note that you do not need to install the CUDA Toolkit on the host system, but the NVIDIA driver needs to be installed. You can learn more at the [NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker) github repo.

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
# replace /path/to/512crop if you're actual path
curl -F '@file=/path/to/512crop.tif' 127.0.0.1:5000/infer_image/0 -o test.pkl
```

## Testing the endpoint
To test your load your numpy array from the pickle file you can do:
```python
import numpy as np
np.load('test.pkl', allow_pickle=True)
```

<!-- references -->
[Argo CD]: https://argoproj.github.io/
[docker]: https://www.docker.com/
[euporie]: https://github.com/joouha/euporie
[External Secrets Operator]: https://external-secrets.io/
[iTerm2]: https://iterm2.com/
[k3s]: https://k3s.io/
[KinD]: https://kind.sigs.k8s.io/
[k9s]: https://k9scli.io/
[minikube]: https://minikube.sigs.k8s.io/docs/
[nvidia drivers]: https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html
[ruff]: https://pypi.org/project/ruff/
[Semshi]: https://github.com/numirias/semshi
[sixel]: https://wikiless.org/wiki/Sixel?lang=en
[smol-k8s-lab]: https://github.com/small-hack/smol-k8s-lab
[w3m]: https://wikiless.org/wiki/W3m?lang=en
[vim]: https://www.vim.org/
[wezterm]: https://wezfurlong.org/wezterm/
[YouCompleteMe]: https://github.com/ycm-core/YouCompleteMe
[zellij]: https://zellij.dev/
