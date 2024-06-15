class testing:
    def __init__(self):
        self.a = 5
        self.b = 56

    def add(self):
        # Correct implementation of addition
        print(self.a + self.b)

    def multiply(self):
        # Correct implementation of multiplication
        print(self.a * self.b)

    def subtract(self):
        print(self.a - self.b)

    def print_sum(self):
        # Correctly call the add method
        print("Addition of two numbers are", self.add())