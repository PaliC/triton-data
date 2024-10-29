import os
import json
import ast
import astor

def find_triton_jit_functions(directories):
    """
    Search through Python files in given directories for functions with @triton.jit decorator
    and return their function bodies as a list of dicts.
    """
    triton_functions = []
    
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            tree = ast.parse(f.read())
                            
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                # Check if function has @triton.jit decorator
                                for decorator in node.decorator_list:
                                    if isinstance(decorator, ast.Attribute):
                                        if (decorator.attr == 'jit' and 
                                            isinstance(decorator.value, ast.Name) and 
                                            decorator.value.id == 'triton'):
                                            # Create a module with just this function
                                            new_module = ast.Module(
                                                body=[node],
                                                type_ignores=[]  # Required for Python 3.8+
                                            )
                                            
                                            # Use astor to convert back to source code
                                            func_body = astor.to_source(new_module)
                                            triton_functions.append({"input": func_body})
                    except Exception as e:
                        print(f"Error processing file {file_path}: {str(e)}")
                        continue
    
    return triton_functions

def main():
    # Get directories from command line arguments
    import sys
    if len(sys.argv) < 2:
        print("Please provide at least one directory path")
        sys.exit(1)
    
    directories = sys.argv[1:]
<<<<<<< HEAD
=======
    print(f"Searching for triton functions in directories: {directories}")
>>>>>>> 3094a4b (hopefully this fixes things)
    
    # Find triton functions and convert to JSON
    triton_funcs = find_triton_jit_functions(directories)
    json_output = json.dumps(triton_funcs, indent=2)
    
<<<<<<< HEAD
    # Output JSON to stdout
    print(json_output)
=======
    print(f"Found {len(triton_funcs)} triton functions")

    # save to file
    with open('datasets/triton_functions.json', 'w') as f:
        f.write(json_output)
>>>>>>> 3094a4b (hopefully this fixes things)

if __name__ == "__main__":
    main()
