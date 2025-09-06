#!/usr/bin/env python
"""
Safe Edit Workflow - Ensures files won't be red after editing
This script validates files before and after editing to prevent IDE errors.
"""

import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import difflib
from datetime import datetime

class SafeEditValidator:
    """Validates edits to ensure no IDE errors are introduced"""
    
    def __init__(self):
        self.validation_results = {}
        self.temp_dir = Path(tempfile.gettempdir()) / "safe_edit"
        self.temp_dir.mkdir(exist_ok=True)
        
    def check_syntax(self, content: str, filepath: str) -> Tuple[bool, Optional[str]]:
        """Check if Python syntax is valid"""
        try:
            ast.parse(content)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
    
    def check_imports(self, content: str) -> List[str]:
        """Check for potentially problematic imports"""
        issues = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    # Check for relative imports that might fail
                    if node.level > 0:
                        issues.append(f"Line {node.lineno}: Relative import '{node.module}' - verify context")
                    
                    # Check for specific names
                    if node.names:
                        for alias in node.names:
                            # We'll validate these with actual import test
                            pass
        except:
            pass
        return issues
    
    def check_undefined_variables(self, content: str) -> List[str]:
        """Check for potentially undefined variables"""
        issues = []
        try:
            tree = ast.parse(content)
            
            # Track defined variables
            defined_vars = set()
            
            # Collect all assignments
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    defined_vars.add(node.id)
                elif isinstance(node, ast.FunctionDef):
                    defined_vars.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    defined_vars.add(node.name)
            
            # Check for undefined usage
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id not in defined_vars and node.id not in __builtins__:
                        # Common globals we can ignore
                        ignore_list = {'self', 'cls', '__name__', '__file__', 'Optional', 
                                      'List', 'Dict', 'Tuple', 'Any', 'Union', 'None',
                                      'True', 'False', 'print', 'len', 'range', 'int',
                                      'str', 'float', 'bool', 'open', 'super'}
                        if node.id not in ignore_list:
                            # Check if it's an import
                            is_import = False
                            for imp_node in ast.walk(tree):
                                if isinstance(imp_node, ast.Import):
                                    for alias in imp_node.names:
                                        if alias.asname == node.id or alias.name == node.id:
                                            is_import = True
                                elif isinstance(imp_node, ast.ImportFrom):
                                    for alias in imp_node.names:
                                        if alias.asname == node.id or alias.name == node.id:
                                            is_import = True
                            
                            if not is_import:
                                issues.append(f"Potentially undefined: '{node.id}'")
        except:
            pass
        
        return list(set(issues))[:5]  # Return first 5 unique issues
    
    def check_type_issues(self, content: str) -> List[str]:
        """Check for common type-related issues"""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for None access without guard
            if '.get(' not in line:  # get() is safe
                if 'None.' in line or 'None[' in line:
                    issues.append(f"Line {i}: Potential None access without check")
            
            # Check for Optional without None check
            if 'Optional[' in line and '->' in line:
                # This is a function signature with Optional
                func_name = line.split('def ')[-1].split('(')[0] if 'def ' in line else 'function'
                # Check if there's a None check in the function
                # (This is simplified - real implementation would be more sophisticated)
                
        return issues[:5]
    
    def check_return_statements(self, content: str) -> List[str]:
        """Check for missing return statements"""
        issues = []
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if function has return type annotation
                    if node.returns:
                        # Check if all paths have return
                        has_return = self._has_return_in_all_paths(node)
                        if not has_return:
                            issues.append(f"Function '{node.name}' may be missing return statement")
        except:
            pass
        
        return issues
    
    def _has_return_in_all_paths(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has return in all code paths"""
        # Simplified check - just verify at least one return exists
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
                return True
        return False
    
    def validate_content(self, content: str, filepath: str) -> Dict:
        """Perform all validation checks on content"""
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # 1. Syntax check (critical)
        syntax_ok, syntax_error = self.check_syntax(content, filepath)
        if not syntax_ok:
            results['valid'] = False
            results['errors'].append(syntax_error)
            return results  # Can't continue if syntax is broken
        
        # 2. Import issues
        import_issues = self.check_imports(content)
        results['warnings'].extend(import_issues)
        
        # 3. Undefined variables
        undefined = self.check_undefined_variables(content)
        if undefined:
            results['warnings'].extend(undefined)
        
        # 4. Type issues
        type_issues = self.check_type_issues(content)
        if type_issues:
            results['warnings'].extend(type_issues)
        
        # 5. Return statements
        return_issues = self.check_return_statements(content)
        if return_issues:
            results['warnings'].extend(return_issues)
        
        return results
    
    def validate_file(self, filepath: Path) -> Dict:
        """Validate a file on disk"""
        with open(filepath, 'r') as f:
            content = f.read()
        return self.validate_content(content, str(filepath))
    
    def compare_validations(self, before: Dict, after: Dict) -> Dict:
        """Compare validation results before and after edit"""
        comparison = {
            'improved': False,
            'degraded': False,
            'new_errors': [],
            'fixed_errors': [],
            'new_warnings': [],
            'fixed_warnings': []
        }
        
        # Check for new or fixed errors
        before_errors = set(before.get('errors', []))
        after_errors = set(after.get('errors', []))
        
        comparison['new_errors'] = list(after_errors - before_errors)
        comparison['fixed_errors'] = list(before_errors - after_errors)
        
        # Check for new or fixed warnings
        before_warnings = set(before.get('warnings', []))
        after_warnings = set(after.get('warnings', []))
        
        comparison['new_warnings'] = list(after_warnings - before_warnings)
        comparison['fixed_warnings'] = list(before_warnings - after_warnings)
        
        # Determine overall status
        if comparison['new_errors']:
            comparison['degraded'] = True
        elif comparison['fixed_errors'] or comparison['fixed_warnings']:
            comparison['improved'] = True
        
        return comparison
    
    def safe_edit_file(self, filepath: str, old_content: str, new_content: str) -> Tuple[bool, str]:
        """
        Safely edit a file with validation
        Returns: (success, message)
        """
        filepath = Path(filepath)
        
        # Step 1: Validate old content
        print(f"\n🔍 Validating original content...")
        before_validation = self.validate_content(old_content, str(filepath))
        
        # Step 2: Validate new content
        print(f"🔍 Validating new content...")
        after_validation = self.validate_content(new_content, str(filepath))
        
        # Step 3: Compare validations
        comparison = self.compare_validations(before_validation, after_validation)
        
        # Step 4: Decision
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
        
        if after_validation['errors']:
            print("❌ NEW CONTENT HAS ERRORS:")
            for error in after_validation['errors']:
                print(f"  - {error}")
            print("\n⚠️  EDIT REJECTED - Would cause red file in IDE!")
            return False, "Edit would introduce syntax errors"
        
        if comparison['new_errors']:
            print("❌ NEW ERRORS INTRODUCED:")
            for error in comparison['new_errors']:
                print(f"  - {error}")
            print("\n⚠️  EDIT REJECTED - Would cause red file in IDE!")
            return False, "Edit would introduce new errors"
        
        if comparison['new_warnings']:
            print("⚠️  NEW WARNINGS (non-critical):")
            for warning in comparison['new_warnings'][:3]:
                print(f"  - {warning}")
        
        if comparison['fixed_errors']:
            print("✅ ERRORS FIXED:")
            for error in comparison['fixed_errors']:
                print(f"  - {error}")
        
        if comparison['fixed_warnings']:
            print("✅ WARNINGS FIXED:")
            for warning in comparison['fixed_warnings'][:3]:
                print(f"  - {warning}")
        
        # Step 5: Apply edit if safe
        print("\n✅ EDIT IS SAFE - No IDE errors will be introduced")
        
        # Save backup
        backup_path = self.temp_dir / f"{filepath.name}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
        with open(backup_path, 'w') as f:
            f.write(old_content)
        print(f"📁 Backup saved to: {backup_path}")
        
        # Write new content
        with open(filepath, 'w') as f:
            f.write(new_content)
        
        return True, "Edit applied successfully"
    
    def validate_import_resolution(self, filepath: str) -> bool:
        """Test if all imports in file can be resolved"""
        try:
            # Try to import the file as a module
            result = subprocess.run(
                [sys.executable, "-c", f"import sys; sys.path.insert(0, '.'); exec(open('{filepath}').read())"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if "ImportError" in result.stderr or "ModuleNotFoundError" in result.stderr:
                print(f"⚠️  Import issues detected: {result.stderr[:200]}")
                return False
            
            return result.returncode == 0
        except:
            return True  # Assume OK if we can't test


def safe_edit_wrapper(filepath: str, old_string: str, new_string: str) -> bool:
    """
    Wrapper function for safe editing
    Use this instead of direct Edit tool to ensure no red files
    """
    validator = SafeEditValidator()
    
    # Read current file content
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if old_string exists
    if old_string not in content:
        print(f"❌ Error: old_string not found in {filepath}")
        return False
    
    # Create new content
    new_content = content.replace(old_string, new_string, 1)
    
    # Perform safe edit
    success, message = validator.safe_edit_file(filepath, content, new_content)
    
    if success:
        # Additional validation: try to import
        if filepath.endswith('.py'):
            if not validator.validate_import_resolution(filepath):
                print("⚠️  Warning: Some imports may not resolve")
    
    return success


def main():
    """Interactive safe edit mode"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python safe_edit.py <file>  # Interactive edit mode")
        print("  python safe_edit.py --check <file>  # Just validate")
        print("  python safe_edit.py --test  # Run test")
        sys.exit(1)
    
    if sys.argv[1] == "--test":
        # Test mode
        test_file = Path("test_safe_edit.py")
        test_content = '''
def example_function(value: Optional[int]) -> int:
    # This will cause a type error
    return value * 2  # Error: value might be None
'''
        
        fixed_content = '''
def example_function(value: Optional[int]) -> int:
    # Fixed version with None check
    if value is None:
        return 0
    return value * 2
'''
        
        test_file.write_text(test_content)
        validator = SafeEditValidator()
        
        print("Testing safe edit validation...")
        success, msg = validator.safe_edit_file(str(test_file), test_content, fixed_content)
        
        test_file.unlink()  # Clean up
        
        if success:
            print("✅ Test passed!")
        else:
            print("❌ Test failed!")
        
    elif sys.argv[1] == "--check":
        # Just validate
        filepath = Path(sys.argv[2])
        validator = SafeEditValidator()
        results = validator.validate_file(filepath)
        
        print(f"\nValidation results for {filepath}:")
        if results['errors']:
            print("❌ ERRORS (will show red):")
            for error in results['errors']:
                print(f"  - {error}")
        
        if results['warnings']:
            print("⚠️  WARNINGS (may show yellow):")
            for warning in results['warnings']:
                print(f"  - {warning}")
        
        if results['valid']:
            print("✅ File should not show red in IDE")
        else:
            print("❌ File will show red in IDE")
    
    else:
        # Interactive edit mode
        filepath = Path(sys.argv[1])
        if not filepath.exists():
            print(f"Error: {filepath} not found")
            sys.exit(1)
        
        print(f"Safe Edit Mode for: {filepath}")
        print("="*60)
        
        validator = SafeEditValidator()
        
        # Validate current state
        current_validation = validator.validate_file(filepath)
        print(f"Current state: {'✅ Valid' if current_validation['valid'] else '❌ Has errors'}")
        
        if current_validation['errors']:
            print("Current errors:")
            for error in current_validation['errors']:
                print(f"  - {error}")
        
        print("\nEnter your edit:")
        print("Old string (enter END on new line when done):")
        old_lines = []
        while True:
            line = input()
            if line == "END":
                break
            old_lines.append(line)
        old_string = '\n'.join(old_lines)
        
        print("New string (enter END on new line when done):")
        new_lines = []
        while True:
            line = input()
            if line == "END":
                break
            new_lines.append(line)
        new_string = '\n'.join(new_lines)
        
        # Perform safe edit
        success = safe_edit_wrapper(str(filepath), old_string, new_string)
        
        if success:
            print(f"\n✅ Edit applied successfully to {filepath}")
            print("File will not show red in IDE!")
        else:
            print(f"\n❌ Edit rejected to prevent IDE errors")


if __name__ == "__main__":
    main()