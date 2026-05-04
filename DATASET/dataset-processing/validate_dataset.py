#!/usr/bin/env python3
"""
Dataset Structural Integrity Validator
Checks JSONL training dataset for structural and content issues
"""

import json
import re
from typing import Dict, List, Tuple
from collections import defaultdict

class DatasetValidator:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.errors = defaultdict(list)
        self.warnings = defaultdict(list)
        self.stats = {
            'total_lines': 0,
            'valid_lines': 0,
            'invalid_json': 0
        }
        
    def validate_json(self, line_num: int, line: str) -> Tuple[bool, dict]:
        """Check if line is valid JSON"""
        try:
            data = json.loads(line.strip())
            return True, data
        except json.JSONDecodeError as e:
            self.errors['json_validity'].append({
                'line': line_num,
                'error': f'Invalid JSON: {str(e)}'
            })
            return False, {}
    
    def validate_keys(self, line_num: int, data: dict) -> bool:
        """Check for presence and non-emptiness of required keys"""
        required_keys = ['instruction', 'input', 'output']
        missing_keys = []
        empty_keys = []
        
        for key in required_keys:
            if key not in data:
                missing_keys.append(key)
            elif not data[key] or (isinstance(data[key], str) and not data[key].strip()):
                empty_keys.append(key)
        
        if missing_keys:
            self.errors['missing_keys'].append({
                'line': line_num,
                'missing': missing_keys
            })
            return False
        
        if empty_keys:
            self.errors['empty_keys'].append({
                'line': line_num,
                'empty': empty_keys
            })
            return False
        
        return True
    
    def validate_input_format(self, line_num: int, input_text: str) -> bool:
        """Check input formatting requirements"""
        issues = []
        
        # Must contain "STUDENT:"
        if "STUDENT:" not in input_text:
            issues.append('Missing "STUDENT:"')
        
        # Must contain "PROBLEM:"
        if "PROBLEM:" not in input_text:
            issues.append('Missing "PROBLEM:"')
        
        if issues:
            self.errors['input_format'].append({
                'line': line_num,
                'issues': issues
            })
            return False
        
        return True
    
    def validate_output_tags(self, line_num: int, output_text: str) -> bool:
        """Validate output tag integrity and order"""
        issues = []
        
        # Define required tags with their patterns
        tags = {
            'internal_solution': (r'<internal_solution>', r'</internal_solution>'),
            'step_analysis': (r'<step_analysis>', r'</step_analysis>'),
            'response': (r'<response>', r'</response>')
        }
        
        # Check each tag opens and closes
        tag_positions = {}
        for tag_name, (open_tag, close_tag) in tags.items():
            open_match = re.search(open_tag, output_text)
            close_match = re.search(close_tag, output_text)
            
            if not open_match:
                issues.append(f'Missing opening tag: {open_tag}')
            if not close_match:
                issues.append(f'Missing closing tag: {close_tag}')
            
            if open_match and close_match:
                if close_match.start() <= open_match.start():
                    issues.append(f'Tag order error: {close_tag} appears before {open_tag}')
                tag_positions[tag_name] = (open_match.start(), close_match.end())
        
        # Check order: internal_solution -> step_analysis -> response
        if len(tag_positions) == 3:
            order = sorted(tag_positions.items(), key=lambda x: x[1][0])
            expected_order = ['internal_solution', 'step_analysis', 'response']
            actual_order = [tag for tag, _ in order]
            
            if actual_order != expected_order:
                issues.append(f'Incorrect tag order. Expected: {expected_order}, Got: {actual_order}')
        
        if issues:
            self.errors['output_tags'].append({
                'line': line_num,
                'issues': issues
            })
            return False
        
        return True
    
    def validate_content_quality(self, line_num: int, data: dict) -> bool:
        """Check content quality metrics"""
        warnings = []
        
        # Check instruction length (should be >= 5 words)
        instruction = data.get('instruction', '')
        instruction_words = len(instruction.split())
        if instruction_words < 5:
            warnings.append(f'Short instruction: only {instruction_words} words')
        
        # Check response content length (inside <response> tags)
        output = data.get('output', '')
        response_match = re.search(r'<response>(.*?)</response>', output, re.DOTALL)
        if response_match:
            response_content = response_match.group(1).strip()
            if len(response_content) < 10:
                warnings.append(f'Short response content: only {len(response_content)} characters')
        
        if warnings:
            self.warnings['content_quality'].append({
                'line': line_num,
                'warnings': warnings
            })
            return False
        
        return True
    
    def run_validation(self) -> Dict:
        """Run all validation checks on the dataset"""
        print(f"Validating dataset: {self.file_path}")
        print("=" * 80)
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                self.stats['total_lines'] += 1
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Step 1: Validate JSON
                is_valid_json, data = self.validate_json(line_num, line)
                if not is_valid_json:
                    self.stats['invalid_json'] += 1
                    continue
                
                # Step 2: Validate required keys
                has_keys = self.validate_keys(line_num, data)
                if not has_keys:
                    continue
                
                # Step 3: Validate input format
                self.validate_input_format(line_num, data['input'])
                
                # Step 4: Validate output tags
                self.validate_output_tags(line_num, data['output'])
                
                # Step 5: Validate content quality
                self.validate_content_quality(line_num, data)
                
                # If we got here without errors, line is valid
                if line_num not in [e['line'] for errors in self.errors.values() for e in errors]:
                    self.stats['valid_lines'] += 1
        
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """Generate a comprehensive validation report"""
        report = {
            'summary': self.stats,
            'errors': dict(self.errors),
            'warnings': dict(self.warnings)
        }
        
        # Calculate additional stats
        report['summary']['error_count'] = sum(len(v) for v in self.errors.values())
        report['summary']['warning_count'] = sum(len(v) for v in self.warnings.values())
        report['summary']['validity_rate'] = (
            self.stats['valid_lines'] / self.stats['total_lines'] * 100 
            if self.stats['total_lines'] > 0 else 0
        )
        
        return report
    
    def print_report(self, report: Dict):
        """Print a formatted validation report"""
        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)
        
        # Summary
        print("\n📊 SUMMARY")
        print("-" * 80)
        print(f"Total lines processed: {report['summary']['total_lines']}")
        print(f"Valid lines: {report['summary']['valid_lines']}")
        print(f"Lines with errors: {report['summary']['total_lines'] - report['summary']['valid_lines']}")
        print(f"Validity rate: {report['summary']['validity_rate']:.2f}%")
        print(f"Total errors: {report['summary']['error_count']}")
        print(f"Total warnings: {report['summary']['warning_count']}")
        
        # Errors by category
        if report['errors']:
            print("\n🚨 ERRORS BY CATEGORY")
            print("-" * 80)
            for category, issues in report['errors'].items():
                print(f"\n{category.upper().replace('_', ' ')} ({len(issues)} issues):")
                # Show first 5 examples
                for issue in issues[:5]:
                    print(f"  Line {issue['line']}: {issue.get('error') or issue.get('issues') or issue.get('missing') or issue.get('empty')}")
                if len(issues) > 5:
                    print(f"  ... and {len(issues) - 5} more")
        
        # Warnings by category
        if report['warnings']:
            print("\n⚠️  WARNINGS BY CATEGORY")
            print("-" * 80)
            for category, issues in report['warnings'].items():
                print(f"\n{category.upper().replace('_', ' ')} ({len(issues)} issues):")
                # Show first 5 examples
                for issue in issues[:5]:
                    print(f"  Line {issue['line']}: {issue.get('warnings')}")
                if len(issues) > 5:
                    print(f"  ... and {len(issues) - 5} more")
        
        # Final verdict
        print("\n" + "=" * 80)
        if report['summary']['error_count'] == 0:
            print("✅ DATASET PASSED ALL VALIDATION CHECKS!")
        else:
            print(f"❌ DATASET HAS {report['summary']['error_count']} ERRORS THAT NEED TO BE FIXED")
        
        if report['summary']['warning_count'] > 0:
            print(f"⚠️  {report['summary']['warning_count']} WARNINGS TO REVIEW")
        
        print("=" * 80)


def main():
    import sys
    
    # Default file path
    file_path = "/Users/lakshmiprajnapenmetsa/AOLM-project/DATASET/final_sft_train_dataset.jsonl"
    
    # Allow command line override
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    # Run validation
    validator = DatasetValidator(file_path)
    report = validator.run_validation()
    validator.print_report(report)
    
    # Save detailed report to file
    report_path = file_path.replace('.jsonl', '_validation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    # Exit with error code if there are errors
    sys.exit(0 if report['summary']['error_count'] == 0 else 1)


if __name__ == "__main__":
    main()
