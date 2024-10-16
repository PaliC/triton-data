import ast
import zipfile
from typing import Set, List, Tuple, Optional, Dict
import os
from pathlib import Path
from dataclasses import dataclass
import tempfile

@dataclass
class TritonKernel:
    """Data class to store information about a Triton kernel"""
    name: str
    file_path: str
    ast_node: ast.FunctionDef
    source_code: str = ""

class TritonSourceGenerator:
    def __init__(self, node: ast.AST, file_content: str, original_path: str):
        """
        Initialize the source generator
        
        Args:
            node: AST node of the Triton function
            file_content: Content of the source file
            original_path: Original path of the file in the zip
        """
        self.node = node
        self.file_content = file_content
        self.file_path = original_path
        self.external_imports: Set[str] = set()
        self.helper_functions: List[ast.FunctionDef] = []
        self.referenced_names: Set[str] = set()
        self.tree = ast.parse(file_content)
        
    def extract_external_imports(self) -> None:
        """Extract all external imports from the file"""
        def format_import(node: ast.AST) -> Optional[str]:
            if isinstance(node, ast.Import):
                names = [
                    f"{alias.name} as {alias.asname}" if alias.asname else alias.name
                    for alias in node.names
                ]
                return f"import {', '.join(names)}" if names else None
                
            elif isinstance(node, ast.ImportFrom):
                names = [
                    f"{alias.name} as {alias.asname}" if alias.asname else alias.name
                    for alias in node.names
                ]
                level = '.' * node.level
                return f"from {level}{node.module} import {', '.join(names)}"
            return None

        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_stmt = format_import(node)
                if import_stmt:
                    self.external_imports.add(import_stmt)
                    
    def find_helper_functions(self) -> None:
        """Find all helper functions referenced by the Triton function"""
        class NameCollector(ast.NodeVisitor):
            def __init__(self):
                self.names = set()
                
            def visit_Name(self, node):
                self.names.add(node.id)
                self.generic_visit(node)
                
        # Collect all referenced names in the Triton function
        collector = NameCollector()
        collector.visit(self.node)
        self.referenced_names = collector.names
        
        # Find helper functions in the same file
        for node in ast.walk(self.tree):
            if (isinstance(node, ast.FunctionDef) and 
                node.name in self.referenced_names and 
                node != self.node):
                self.helper_functions.append(node)
                
                # Recursively check helper functions for more references
                collector = NameCollector()
                collector.visit(node)
                self.referenced_names.update(collector.names)
                
    def get_decorators(self) -> List[str]:
        """Extract decorators from the function node"""
        decorators = []
        for decorator in self.node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(f"@{decorator.id}")
            elif isinstance(decorator, ast.Attribute):
                decorators.append(f"@{ast.unparse(decorator)}")
            elif isinstance(decorator, ast.Call):
                decorators.append(f"@{ast.unparse(decorator)}")
        return decorators
        
    def generate_source(self) -> str:
        """Generate the complete source code"""
        # Extract imports and helper functions
        self.extract_external_imports()
        self.find_helper_functions()
        
        # Sort imports
        std_lib_imports = set()
        third_party_imports = set()
        
        for imp in self.external_imports:
            module = imp.split()[1].split('.')[0]
            try:
                module_path = __import__(module).__file__
                if module_path and ('site-packages' in module_path or 
                                  'dist-packages' in module_path):
                    third_party_imports.add(imp)
                else:
                    std_lib_imports.add(imp)
            except (ImportError, AttributeError):
                third_party_imports.add(imp)
                
        # Build the source code
        source_parts = [
            f"# Generated from {self.file_path}",
            f"# Original function: {self.node.name}\n"
        ]
        
        # Add imports
        if std_lib_imports:
            source_parts.extend([
                "# Standard library imports",
                "\n".join(sorted(std_lib_imports)),
                ""
            ])
            
        if third_party_imports:
            source_parts.extend([
                "# Third-party imports",
                "\n".join(sorted(third_party_imports)),
                ""
            ])
            
        # Add helper functions
        if self.helper_functions:
            source_parts.extend([
                "# Helper functions",
                "\n\n".join(ast.unparse(func) for func in self.helper_functions),
                ""
            ])
            
        # Add main function with decorators
        source_parts.extend([
            "# Triton kernel function",
            "\n".join(self.get_decorators()),
            ast.unparse(self.node)
        ])
        
        return "\n\n".join(source_parts)

def find_triton_kernels(zip_path: str) -> List[TritonKernel]:
    """
    Find all Triton kernels in a zip file
    
    Args:
        zip_path: Path to the zip file
        
    Returns:
        List[TritonKernel]: List of found Triton kernels
    """
    kernels = []
    
    def is_triton_decorator(node: ast.AST) -> bool:
        """Check if a node is a triton.jit decorator"""
        if isinstance(node, ast.Name) and node.id == 'jit':
            return True
        elif isinstance(node, ast.Attribute):
            return (isinstance(node.value, ast.Name) and 
                   node.value.id == 'triton' and 
                   node.attr == 'jit')
        return False
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.filelist:
            if not file_info.filename.endswith('.py'):
                continue
                
            try:
                with zip_ref.open(file_info.filename) as f:
                    content = f.read().decode('utf-8')
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if (isinstance(node, ast.FunctionDef) and 
                            any(is_triton_decorator(d) for d in node.decorator_list)):
                            kernel = TritonKernel(
                                name=node.name,
                                file_path=file_info.filename,
                                ast_node=node
                            )
                            kernels.append(kernel)
                            
            except Exception as e:
                print(f"Error processing {file_info.filename}: {str(e)}")
                continue
                
    return kernels

def generate_kernel_sources(zip_path: str, output_dir: str) -> Dict[str, str]:
    """
    Process a zip file and generate source code for all Triton kernels
    Args:
        zip_path: Path to the zip file
        output_dir: Directory to save generated source files   
    Returns:
        Dict[str, str]: Mapping of kernel names to their file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Find all kernels
    kernels = find_triton_kernels(zip_path)
    print(f"Found {len(kernels)} kernels")
    # Generate source for each kernel
    kernel_files = {}
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for kernel in kernels:
            try:
                # Read the original file content
                with zip_ref.open(kernel.file_path) as f:
                    content = f.read().decode('utf-8')
                # Generate source code
                generator = TritonSourceGenerator(
                    kernel.ast_node,
                    content,
                    kernel.file_path
                )
                source_code = generator.generate_source() 
                # Save to file
                output_file = os.path.join(
                    output_dir,
                    f"{kernel.name}_kernel.py"
                )
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(source_code)                   
                    # print location of file
                    print(f"Saved to {output_file}")
                kernel_files[kernel.name] = output_file               
            except Exception as e:
                print(f"Error generating source for {kernel.name}: {str(e)}")
                continue               
    return kernel_files

if __name__ == "__main__":
    zip_path = "/Users/sahanp/repos/liger.zip"
    output_dir = "/Users/sahanp/triton-data/scraper/triton-kernels"
    generate_kernel_sources(zip_path, output_dir)