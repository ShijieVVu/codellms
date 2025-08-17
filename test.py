from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    """
    Input to this function is a string containing multiple groups of nested parentheses.
    Your goal is to separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    # TODO: Write your code here
    count = 0
    stack = []
    result = []
    for char in paren_string:
        if char == '(':
            stack.append(char)
            count += 1
        elif char == ')':
            if len(stack) == 0:
                return []
            else:
                stack.append(char)
                count -= 1
                if count == 0:
                    result.append(''.join(stack))
                    stack = []
    if len(stack) != 0:
        return []
    return result


if __name__ == '__main__':
    print(separate_paren_groups('( ) (( )) (( )( ))'))