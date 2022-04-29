import kfp
import kfp.components as comp
import kfp.dsl as dsl



create_step_get_lines = comp.load_component_from_text("""
name: Get Lines
description: Gets the specified number of lines from the input file.

inputs:
- {name: input_1, type: String, description: 'Data for input_1'}
- {name: input_2, type: String, description: 'Data for input_2'}


implementation:
  container:
    image: zhuyaguang/pipeline:v4
    # command is a list of strings (command-line arguments). 
    # The YAML language has two syntaxes for lists and you can use either of them. 
    # Here we use the "flow syntax" - comma-separated strings inside square brackets.
    command: [
      python3, 
      # Path of the program inside the container
      /pipelines/component/src/testdemo.py,
      --config,
      {inputValue: input_1},
      --model, 
      {inputValue: input_2},
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
    name="example2-pipeline",
) 
def my_pipeline():
    get_lines_step = create_step_get_lines(
        # Input name "Input 1" is converted to pythonic parameter name "input_1"
        input_1='hangzhou',
        input_2='shanghai',
    )

if __name__ == '__main__':
    # Compiling the pipeline
    kfp.compiler.Compiler().compile(my_pipeline, 'v3.yaml')