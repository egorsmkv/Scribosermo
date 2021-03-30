# Testing

Build container with testing tools:

```bash
podman build -f tests/Containerfile -t testing_scribosermo ./Scribosermo/

podman run --network host --rm \
  --volume `pwd`/Scribosermo/:/Scribosermo/ \
  -it testing_scribosermo
```

Execute unit tests:

```bash
# Run in container
cd /Scribosermo/ && pytest --cov=preprocessing
```

For syntax tests, check out the steps in the `gitlab-ci.yml` file.
