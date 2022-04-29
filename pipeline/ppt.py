#!/usr/bin/env python
# coding: utf-8


import kfp
from kfp import compiler
import kfp.dsl as dsl
import kfp.notebook
import kfp.components as comp



#Define a Python function
def DataCollection(a: float, b: float) -> float:
    '''Calculates sum of two arguments'''
    return a + b


add_op = comp.func_to_container_op(DataCollection)



def DataMining(a: float, b: float)-> float:
    return a+b

divmod_op = comp.func_to_container_op(DataMining)

def PretrainedModel_ST_BERT(
    a: float, b: float
) -> float:
    return a+b

pm_op = comp.func_to_container_op(PretrainedModel_ST_BERT)


def KnowledgeExpression(a: float, b: float) -> float:
    
    return a + b

ke_op = comp.func_to_container_op(KnowledgeExpression)

def ModelServing(a: float, b: float) -> float:
    
    return a + b

ms_op = comp.func_to_container_op(ModelServing)

def PlatformBuilding(a: float, b: float) -> float:
    
    return a + b

pb_op = comp.func_to_container_op(PlatformBuilding)

@dsl.pipeline(
    name='Calculation pipeline',
    description='A toy pipeline that performs arithmetic calculations.')
def calc_pipeline(
    a='a',
    b='7',
    c='17',
):
    #Passing pipeline parameter and a constant value as operation arguments
    add_task = add_op(a, 4)  # Returns a dsl.ContainerOp class instance.

    #Passing a task output reference as operation arguments
    #For an operation with a single return value, the output reference can be accessed using `task.output` or `task.outputs['output_name']` syntax
    divmod_task = divmod_op(add_task.output, b)

    #For an operation with a multiple return values, the output references can be accessed using `task.outputs['output_name']` syntax
    result_task = pm_op(divmod_task.output, c)

    ke_task = ke_op(result_task.output, c)

    ms_task = ms_op(ke_task.output, c)

    pb_task = pb_op(ms_task.output, c)


if __name__ == '__main__':
    # Compiling the pipeline
    kfp.compiler.Compiler().compile(calc_pipeline, 'ppt.yaml')