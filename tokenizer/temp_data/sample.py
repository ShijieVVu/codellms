
def hello_world():
    print("Hello, World!")
    return True

class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b

# Main execution
if __name__ == "__main__":
    calc = Calculator()
    result = calc.add(5, 3)
    print(f"5 + 3 = {result}")
