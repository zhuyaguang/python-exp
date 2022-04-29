import kfp
import kfp.components as comp
import kfp.dsl as dsl
create_step_get_lines = comp.load_component_from_text("""
name: Get Lines sunhao
description: lalallalala.
implementation:
  container:
    image: 10.100.29.62/kubeflow/train:v3
    # command is a list of strings (command-line arguments). 
    # The YAML language has two syntaxes for lists and you can use either of them. 
    # Here we use the "flow syntax" - comma-separated strings inside square brackets.
    command: [
      python3, 
      # Path of the program inside the container
      /home/pipeline-demo/train.py,
    ]""")

# Define your pipeline
@dsl.pipeline(
    pipeline_root='',
    name="example-pipeline",
) 
def my_pipeline():
    get_lines_step = create_step_get_lines(
    )

if __name__ == '__main__':
    # Compiling the pipeline
    kfp.compiler.Compiler().compile(my_pipeline, 'sunhao3.yaml')