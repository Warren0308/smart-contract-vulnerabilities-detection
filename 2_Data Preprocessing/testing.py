import re
import string

def preprocess_source_code(source_code):
    # Remove multi-line comments (/* ... */)
    code = re.sub(r'/\*.*?\*/', '', source_code, flags=re.DOTALL)

    # Remove single-line comments (// ...)
    code = re.sub(r'//.*', '', code)

    # Remove newline characters and excessive whitespace
    code = re.sub(r'\n', ' ', code)
    code = re.sub(r'\s+', ' ', code).strip()
    code = code.replace(string.punctuation,'')
    # Regex pattern to match 'pragma solidity ^version;'
    code = re.sub(r'pragma solidity\s+\^?[0-9]+\.[0-9]+\.[0-9 ]+;', '', code)

    return code


# Example usage
original_source_code = """
pragma solidity ^0.8.0;

// This is a test contract
contract TestContract {
    uint256 public value = 100;

    function setValue(uint256 newValue) public {
        value = newValue; // Assign new value
    }
}
"""

processed_code = preprocess_source_code(original_source_code)
print(processed_code)
