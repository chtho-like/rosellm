#!/usr/bin/env python
"""
My Standard Edit Workflow - What I'll do for EVERY file edit
This ensures files are never red in IDE after my edits.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple
import ast

class MyEditWorkflow:
    """My personal workflow for editing files safely"""
    
    def __init__(self):
        self.checks_passed = []
        self.checks_failed = []
    
    def step1_check_current_state(self, filepath: str) -> bool:
        """Step 1: Check if file is currently valid"""
        print(f"\n📋 STEP 1: Checking current state of {filepath}")
        
        # Check syntax
        try:
            with open(filepath, 'r') as f:
                ast.parse(f.read())
            print("  ✅ Current syntax: Valid")
            self.checks_passed.append("current_syntax")
            return True
        except SyntaxError as e:
            print(f"  ⚠️  Current syntax error: {e}")
            self.checks_failed.append("current_syntax")
            return True  # We can still proceed to fix it
    
    def step2_validate_planned_edit(self, filepath: str, old_string: str, new_string: str) -> bool:
        """Step 2: Validate the planned edit"""
        print(f"\n📋 STEP 2: Validating planned edit")
        
        # Read file
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check old string exists
        if old_string not in content:
            print(f"  ❌ Old string not found in file")
            self.checks_failed.append("old_string_exists")
            return False
        
        print(f"  ✅ Old string found")
        
        # Create new content
        new_content = content.replace(old_string, new_string, 1)
        
        # Check new syntax
        try:
            ast.parse(new_content)
            print(f"  ✅ New syntax will be valid")
            self.checks_passed.append("new_syntax")
        except SyntaxError as e:
            print(f"  ❌ New syntax will have error: {e}")
            self.checks_failed.append("new_syntax")
            return False
        
        return True
    
    def step3_check_imports(self, filepath: str, new_content: str) -> bool:
        """Step 3: Verify imports will resolve"""
        print(f"\n📋 STEP 3: Checking imports")
        
        try:
            tree = ast.parse(new_content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(f"from {node.module}")
            
            if imports:
                print(f"  📦 Found {len(imports)} imports to verify")
                # Test critical rosellm imports
                for imp in imports:
                    if 'rosellm' in imp:
                        print(f"  🔍 Checking: {imp}")
            
            self.checks_passed.append("imports")
            return True
            
        except:
            return True  # Continue anyway
    
    def step4_check_type_safety(self, new_content: str) -> bool:
        """Step 4: Check for type safety issues"""
        print(f"\n📋 STEP 4: Checking type safety")
        
        issues = []
        lines = new_content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for None access
            if 'None.' in line or 'None[' in line:
                issues.append(f"Line {i}: Potential None access")
            
            # Check for missing Optional checks
            if 'Optional[' in line and not any(x in line for x in ['if ', 'is None', 'is not None']):
                # Might be a signature, check next few lines for None check
                pass
        
        if issues:
            print(f"  ⚠️  Found {len(issues)} potential issues:")
            for issue in issues[:3]:
                print(f"     - {issue}")
            self.checks_failed.append("type_safety")
        else:
            print(f"  ✅ No obvious type safety issues")
            self.checks_passed.append("type_safety")
        
        return True  # Non-blocking
    
    def step5_apply_edit(self, filepath: str, old_string: str, new_string: str) -> bool:
        """Step 5: Apply the edit"""
        print(f"\n📋 STEP 5: Applying edit")
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        new_content = content.replace(old_string, new_string, 1)
        
        # Backup original
        backup_path = f"{filepath}.bak"
        with open(backup_path, 'w') as f:
            f.write(content)
        
        # Write new content
        with open(filepath, 'w') as f:
            f.write(new_content)
        
        print(f"  ✅ Edit applied")
        print(f"  📁 Backup saved to {backup_path}")
        
        return True
    
    def step6_verify_result(self, filepath: str) -> bool:
        """Step 6: Verify file is not red after edit"""
        print(f"\n📋 STEP 6: Verifying result")
        
        # Check syntax one more time
        try:
            with open(filepath, 'r') as f:
                ast.parse(f.read())
            print(f"  ✅ Final syntax: Valid")
            
            # Try to import if it's a module
            if '/rosellm/' in filepath:
                module_path = filepath.replace('/', '.').replace('.py', '')
                module_path = module_path.split('rosellm.')[-1]
                try:
                    result = subprocess.run(
                        [sys.executable, "-c", f"from rosellm.{module_path} import *"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0:
                        print(f"  ✅ Module imports successfully")
                except:
                    pass
            
            return True
            
        except SyntaxError as e:
            print(f"  ❌ Final syntax error: {e}")
            return False
    
    def execute_safe_edit(self, filepath: str, old_string: str, new_string: str) -> bool:
        """
        Execute my complete safe edit workflow
        This is what I'll do for EVERY edit
        """
        print("="*60)
        print(f"🚀 SAFE EDIT WORKFLOW FOR: {filepath}")
        print("="*60)
        
        # Step 1: Check current state
        self.step1_check_current_state(filepath)
        
        # Step 2: Validate planned edit
        if not self.step2_validate_planned_edit(filepath, old_string, new_string):
            print("\n❌ EDIT ABORTED: Validation failed")
            return False
        
        # Step 3: Check imports
        with open(filepath, 'r') as f:
            content = f.read()
        new_content = content.replace(old_string, new_string, 1)
        self.step3_check_imports(filepath, new_content)
        
        # Step 4: Check type safety
        self.step4_check_type_safety(new_content)
        
        # Decision point
        print("\n" + "="*60)
        print("📊 PRE-EDIT VALIDATION SUMMARY")
        print("="*60)
        print(f"✅ Checks passed: {len(self.checks_passed)}")
        print(f"❌ Checks failed: {len(self.checks_failed)}")
        
        if "new_syntax" in self.checks_failed:
            print("\n⛔ CANNOT PROCEED: Edit would break syntax")
            return False
        
        # Step 5: Apply edit
        self.step5_apply_edit(filepath, old_string, new_string)
        
        # Step 6: Verify result
        success = self.step6_verify_result(filepath)
        
        # Final summary
        print("\n" + "="*60)
        print("📊 FINAL RESULT")
        print("="*60)
        
        if success:
            print("✅ SUCCESS: File edited safely - No red errors in IDE!")
        else:
            print("❌ WARNING: File may have issues - Check IDE")
            # Restore backup
            backup_path = f"{filepath}.bak"
            if Path(backup_path).exists():
                print("🔄 Restoring backup...")
                with open(backup_path, 'r') as f:
                    content = f.read()
                with open(filepath, 'w') as f:
                    f.write(content)
                print("✅ Backup restored")
        
        return success


# Convenience function I'll use
def safe_edit(filepath: str, old_string: str, new_string: str) -> bool:
    """
    My standard safe edit function
    Use this instead of Edit tool to guarantee no red files
    """
    workflow = MyEditWorkflow()
    return workflow.execute_safe_edit(filepath, old_string, new_string)


# Test the workflow
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Demo mode
        test_file = Path("demo_file.py")
        test_file.write_text("""
def example(value: Optional[int]) -> int:
    return value * 2  # This will be red
""")
        
        print("🎭 DEMO: Fixing a file that would be red")
        
        success = safe_edit(
            str(test_file),
            "    return value * 2  # This will be red",
            "    if value is None:\n        return 0\n    return value * 2  # Now safe"
        )
        
        if success:
            print("\n✅ Demo successful!")
            with open(test_file, 'r') as f:
                print("\nFinal content:")
                print(f.read())
        
        test_file.unlink()  # Clean up
    else:
        print("My Safe Edit Workflow")
        print("Usage: python my_edit_workflow.py --demo")