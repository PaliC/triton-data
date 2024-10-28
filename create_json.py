import os
import json
import ast

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
                                            
                                            # Get the function source code
                                            start_lineno = node.lineno
                                            end_lineno = node.end_lineno
                                            with open(file_path, 'r') as f:
                                                lines = f.readlines()
                                                func_body = ''.join(lines[start_lineno-1:end_lineno])
                                                
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
    
    # Find triton functions and convert to JSON
    triton_funcs = find_triton_jit_functions(directories)
    json_output = json.dumps(triton_funcs, indent=2)
    
    # Output JSON to stdout
    print(json_output)

if __name__ == "__main__":
    main()
