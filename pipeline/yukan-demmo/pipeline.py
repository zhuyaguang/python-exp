from ast import arguments
from unicodedata import name

from setuptools import Command
from kfp import dsl, compiler

    
def yukan_op():
    return dsl.ContainerOp(
        name='yukan-train',
        image='10.100.29.62/kubeflow/train-bert:v3',
        command=['python3', '/home/pipeline-demo/yukan-demo/train.py'],

    )

@dsl.pipeline(
    name='sunhao-pipeline',
    description='A pipeline with two sequential steps.'
)
def sequential_pipeline():
    """A pipeline with two sequential steps."""

    download_task = yukan_op()
   
if __name__ == '__main__':
    compiler.Compiler().compile(sequential_pipeline, 'yk.yaml')