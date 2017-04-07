#!/bin/bash
# set default environment variables

set -e

WITH_CMAKE=${WITH_CMAKE:-false}
WITH_PYTHON3=${WITH_PYTHON3:-false}
WITH_IO=${WITH_IO:-true}
WITH_CUDA=${WITH_CUDA:-false}
WITH_CUDNN=${WITH_CUDNN:-false}
