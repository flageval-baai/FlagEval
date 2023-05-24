import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 环境
#
# - development
# - prod
ENV = "development"

PROJECT_NAME = "FlagEval"

DEFAULT_PROJECT_NAME = PROJECT_NAME

# Application definition
INSTALLED_APPS = [
    "flageval.serving.service",
]
