# Contributing guidelines

## Style

#### C++ coding style

Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html), with max-line-length extended to 100.

Run `clang-format` before committing code:

```
clang-format -i <my_cc_file>
```

#### Python coding style

Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with exceptions:

* max-line-length = 100
* indent_width = 2
* Specify argument type in bracket in docstring. For example:

```python
"""USE
Args:
  embedding_width (int): Width of embedding vector
"""

"""DONOT use
Args:
  embedding_width: An `integer`. Width of embedding vector.
"""
```

Run `pylint` before committing code.

Install `pylint`

```bash
pip install pylint
```

To check a file with `pylint`:

```bash
pylint --rcfile=.pylintrc myfile.py
```

#### Yapf

[yapf](https://github.com/google/yapf/) is an auto format tool owned by Google (not a Google product). To save the time of arguing code style during code review, use yapf to format the code is a good option. Note that it doesn't reformat comment.

Install `yapf`

```bash
pip install yapf
```

Format code with yapf

```bash
yapf myfile.py --style .style.yapf
```

There are Sublime and Vim plugins.

## Test

Use [googletest](https://github.com/google/googletest) for c++ code.

Use Tensorflow's test and [abseil](https://github.com/abseil/abseil-py) for python code.

DONOT use mixed case test name, follow the code style.

## Miscellaneous

### Run pre-commit

[pre-commit](https://github.com/pre-commit/pre-commit) manages pre git commit hooks. `.pre-commit-config.yaml` is configured to run coding style and some other checks.

### Use abseil

In addition to test, abseil also provides `flags` and `logging` which is derived from Tensorflow. We recommend to abseil flags and logging over Python's argparse and logging.

## Developer Certificate of Origin (DCO)

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
