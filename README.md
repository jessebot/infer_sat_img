# Getting started locally

## Testing the app on K8s locally

```bash
```

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

### testing `infer_image`

```bash
curl -F '@file=/home/USER/file.tiff' 127.0.0.1:5000/infer_image/0 -o test.pkl
```

To test your load your numpy array from the pickle file you can do:
```python
import numpy as np
np.load('test.pkl', allow_pickle=True)
```
