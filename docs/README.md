# Documentation

This folder contains the scripts necessary to build the documentation.

## Building the documentation

1. To build the docs, start a container and mounting the root directory
   of the repository into the container at the `/workspace` mount point:

   ```shell
   docker run --rm -it --net=host \
     -u $(id -u):$(id -g) -v $(pwd):/workspace -w /workspace \
     nvcr.io/nvidia/merlin/merlin-tensorflow:nightly
   ```

1. Build the software:

   ```shell
   make all
   ```

1. Install required documentation tools and extensions:

   ```shell
   export HOME=/tmp
   export PATH=$HOME/.local/bin:$PATH
   python -m pip install -r docs/requirements-doc.txt
   ```

1. Build the documentation:

   ```shell
   export PYTHONPATH=$(python -c 'import site; print(site.getusersitepackages(), end="")')
   make -C docs clean html
   ```

   > **Troubleshooting Tip** To get verbose output from the Sphinx build, run
   > `make -C docs clean` and then run `sphinx-build -vv docs/source docs/build/html`.
   > When Sphinx loads libraries, the verbose output shows the path for the library.

The preceding command runs Sphinx in your shell and outputs to `docs/build/html/`.

In a shell that is outside the container, start a simple HTTP server:

`python -m http.server -d docs/build/html 8000`

Then, navigate a web browser to the IP address or hostname of the host machine at port 8000:

`https://localhost:8000`

Now you can check if your docs edits formatted correctly, and read well.

