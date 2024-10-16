import zipfile
import re
import ast
import argparse
import astor

def extract_triton_jit_functions(zip_file_path):
    triton_jit_functions = []
    triton_function_nodes = []

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.py'):
                with zip_ref.open(file_name) as file:
                    content = file.read().decode('utf-8')
                    
                    # Use AST to parse the Python code
                    try:
                        tree = ast.parse(content)
                    except SyntaxError:
                        # Skip files with syntax errors
                        continue

                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            for decorator in node.decorator_list:
                                if isinstance(decorator, ast.Attribute):
                                    if (isinstance(decorator.value, ast.Name) and decorator.value.id == 'triton' and decorator.attr == 'jit'):
                                        triton_jit_functions.append((file_name, node.name))
                                        triton_function_nodes.append(node)
                                elif isinstance(decorator, ast.Name) and decorator.id == 'jit':
                                    # Handle case where triton.jit might be imported directly
                                    triton_jit_functions.append((file_name, node.name))
                                    triton_function_nodes.append(node)
    # Function to generate runnable code for a given AST node
    def generate_runnable_code(node):
        # Import statements
        imports = [
            "import triton",
            "import triton.language as tl",
            "import torch",  # Commonly used with Triton, include just in case
        ]
        # Generate the function code
        function_code = astor.to_source(node)
        # Combine imports and function code
        runnable_code = "\n".join(imports) + "\n\n" + function_code
        
        # Try to compile the code to ensure it's runnable
        try:
            compile(runnable_code, '<string>', 'exec')
        except Exception as e:
            print(f"Error compiling code for function '{node.name}': {str(e)}")
            return None
        
        return runnable_code

    runnable_functions = [(node.name, generate_runnable_code(node)) for node in triton_function_nodes]

    return triton_jit_functions
    return runnable_functions





# Example usage:

if __name__ == '__main__':
    
    # Path to the zip file is arg 1
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_file_path", help="Path to the zip file")
    args = parser.parse_args()

    functions = extract_triton_jit_functions(args.zip_file_path)
    for file_name, func_name in functions:
        print(f"File: {file_name}, Function: {func_name}")
