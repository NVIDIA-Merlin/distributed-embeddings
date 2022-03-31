#!/usr/bin/env bash
set -e
set -x

mkdir -p artifacts

TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)
echo $(date) : "=== Using tmpdir: ${TMPDIR}"
echo "=== Copy TensorFlow Custom op files"

cp setup.py "${TMPDIR}"
cp MANIFEST.in "${TMPDIR}"
cp requirements.txt "${TMPDIR}"
rsync -avm -L --exclude='*_test.py' distributed_embeddings "${TMPDIR}"

pushd ${TMPDIR}
echo $(date) : "=== Building wheel"

python3 setup.py bdist_wheel > /dev/null

popd
cp ${TMPDIR}/dist/*.whl artifacts
rm -rf ${TMPDIR}
