import kfp
import kfp.components as comp
import kfp.dsl as dsl
create_step_get_lines = comp.load_component_from_text("""
name: Train data by sunhao
description: train 300 lines data 

inputs:
- {name: input_1, type: String, description: '--config'}
- {name: input_2, type: String, description: '--model'}
- {name: input_3, type: String, description: '--file_path'}
- {name: parameter_1, type: Integer, default: '-1', description: 'For distributed training: local_rank'}

outputs:
- {name: output_1, type: String, description: 'save_dir .'}

implementation:
  container:
    image: 10.100.29.62:30080/kubeflow/zhuyaguang/pipeline:v6
    # command is a list of strings (command-line arguments). 
    # The YAML language has two syntaxes for lists and you can use either of them. 
    # Here we use the "flow syntax" - comma-separated strings inside square brackets.
    command: [
      python3, 
      # Path of the program inside the container
      /home/pipeline-demo/train.py,
      --config,
      {inputPath: input_1},
      --model,
      {inputPath: input_2},
      --file_path,
      {inputPath: input_3},
      --local_rank, 
      {inputValue: parameter_1},
      --save_dir, 
      {outputPath: output_1},
    ]""")

# create_step_get_lines is a "factory function" that accepts the arguments
# for the component's inputs and output paths and returns a pipeline step
# (ContainerOp instance).
#
# To inspect the get_lines_op function in Jupyter Notebook, enter 
# "get_lines_op(" in a cell and press Shift+Tab.
# You can also get help by entering `help(get_lines_op)`, `get_lines_op?`,
# or `get_lines_op??`.

# Define your pipeline
@dsl.pipeline(
    pipeline_root='',
    name="train-pipeline",
) 
def my_pipeline():
    get_lines_step = create_step_get_lines(
        # Input name "Input 1" is converted to pythonic parameter name "input_1"
        input_1='bert-base-uncased',
        parameter_1='-1',
    )

if __name__ == '__main__':
    # Compiling the pipeline
    kfp.compiler.Compiler().compile(my_pipeline, 'v6.yaml')