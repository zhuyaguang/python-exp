#!/usr/bin/env python3
# Copyright 2019 The Kubeflow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from ast import arguments
from unicodedata import name

from setuptools import Command
from kfp import dsl, compiler

    
def sunhao_op():
    return dsl.ContainerOp(
        name='sunhao-train',
        image='10.100.29.62/kubeflow/train:v3',
        command=['python3', '/home/pipeline-demo/train.py'],
        file_outputs={
            'data': '/home/pipeline-demo/config.json',
        }
    )

def sunhao2_op():
    return dsl.ContainerOp(
        name='sunhao2-train',
        image='10.100.29.62/kubeflow/train:v3',
        command=['sh', '-c'],
        arguments=['python3 /home/pipeline-demo/train.py']
    )

@dsl.pipeline(
    name='sunhao-pipeline',
    description='A pipeline with two sequential steps.'
)
def sequential_pipeline():
    """A pipeline with two sequential steps."""

    download_task = sunhao_op()
    download_task2 = sunhao2_op()

if __name__ == '__main__':
    compiler.Compiler().compile(sequential_pipeline, 'sh.yaml')