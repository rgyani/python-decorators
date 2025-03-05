# Python Decorators

A decorator in Python is a function that takes another function (or class) as input and extends or alters its behavior without modifying its structure. Decorators are commonly used to add functionality such as logging, error handling, authentication, or performance measurement to functions or methods in a clean and reusable way.


They are implemented using the @decorator_name syntax.

### How Decorators Work
A decorator is a higher-order function that takes a function as input and returns a new function with added behavior.
The decorated function is essentially "wrapped" by the decorator.

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        # Add behavior before the original function
        print("Before the function call")
        result = func(*args, **kwargs)
        # Add behavior after the original function
        print("After the function call")
        return result
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```

Output:
```
Before the function call
Hello!
After the function call
```

## Using Decorators in ETL Processes
In ETL (Extract, Transform, Load) processes, decorators can help make code reusable and modular. Common use cases include **logging, error handling, performance monitoring, and retry mechanisms.**

#### Example Use Cases
1. **Logging** Track the start, end, and duration of ETL tasks.
```python
import time

def log_etl_step(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Starting: {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Finished: {func.__name__} in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@log_etl_step
def extract_data():
    time.sleep(2)  # Simulate data extraction
    print("Extracted data")

@log_etl_step
def transform_data():
    time.sleep(1)  # Simulate data transformation
    print("Transformed data")

@log_etl_step
def load_data():
    time.sleep(3)  # Simulate data loading
    print("Loaded data")

# Run the ETL steps
extract_data()
transform_data()
load_data()
```

2. **Error Handling** Automatically retry ETL steps in case of failure.
```python
import time
import random

def retry_on_failure(retries=3, delay=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error: {e}, retrying {attempt + 1}/{retries}...")
                    time.sleep(delay)
            raise Exception(f"Failed after {retries} retries")
        return wrapper
    return decorator

@retry_on_failure(retries=3, delay=1)
def load_data():
    # Simulate a process that sometimes fails
    if random.random() < 0.5:
        raise ValueError("Random failure during loading")
    print("Loaded data successfully")

# Run the ETL step
load_data()
```

3. **Performance Monitoring** Measure and log the time taken by each ETL step.
```python
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print(f"{func.__name__} took {duration:.2f} seconds")
        return result
    return wrapper

@measure_time
def transform_data():
    time.sleep(2)  # Simulate a complex transformation
    print("Data transformed")

# Run the transformation step
transform_data()
```

4. **Reusability and Modularity** Combine multiple decorators to enhance ETL functions.
```python
@log_etl_step
@retry_on_failure(retries=3, delay=1)
@measure_time
def load_data():
    if random.random() < 0.5:
        raise ValueError("Random failure during loading")
    print("Loaded data successfully")

# Run the enhanced ETL function
load_data()
```

#### Benefits of Using Decorators in ETL Processes
1. **Code Reusability:** Decorators encapsulate reusable functionality, reducing code duplication.
2. **Modularity:** Keep the core ETL logic separate from auxiliary features like logging or error handling.
3. **Readability:** By using the @decorator_name syntax, the purpose of the function is clear.
4. **Maintainability:** Changes to shared functionality (e.g., logging) only need to be updated in the decorator.

By using decorators, you can optimize and standardize ETL pipelines, making them more robust and easier to manage.


## More Decorator Patterns

#### 1. @property - Clean Attribute Access
The @property decorator transforms methods into attributes, allowing you to add validation logic while maintaining a clean interface.

```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value

    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9
```

#### 2. @functools.cached_property - Lazy Computed Properties
 
The @cached_property decorator from the functools module combines @property with caching, computing a value only once and then storing it until the instance is deleted.

```python
from functools import cached_property
import time

class DataAnalyzer:
    def __init__(self, dataset):
        self.dataset = dataset

    @cached_property
    def complex_analysis(self):
        print("Running expensive analysis...")
        time.sleep(2)  # Simulating heavy computation
        return sum(x**2 for x in self.dataset)
```

#### 3. @functools.lru_cache - Memoization
 
The lru_cache decorator caches function results based on arguments. Which helps in speeding up expensive calculations.

Here’s a recursive Fibonacci sequence calculator that would normally be inefficient. But the @lru_cache decorator makes it by storing previously calculated values.

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

#### 4. @contextlib.contextmanager - Custom Context Managers
 
The contextmanager decorator from contextlib lets you create your own context managers without implementing the full __enter__/__exit__ protocol.

Let’s see how to create custom context managers with minimal code. The file_manager function ensures files are properly closed even if exceptions occur:
```python
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
	try:
    	f = open(filename, mode)
    	yield f
	finally:
    	f.close() 
```

#### 5. @functools.singledispatch - Function Overloading
 
The @singledispatch decorator from functools implements single-dispatch generic functions, allowing different implementations based on argument type.

Let’s create a flexible formatting system that handles different data types appropriately. The @singledispatch decorator lets you define a default implementation for the format_output function, then register specialized versions for different types.

```python
from functools import singledispatch
from datetime import date, datetime

@singledispatch
def format_output(obj):
    return str(obj)

@format_output.register
def _(obj: int):
    return f"INTEGER: {obj:+d}"

@format_output.register
def _(obj: float):
    return f"FLOAT: {obj:.2f}"

@format_output.register
def _(obj: date):
    return f"DATE: {obj.strftime('%Y-%m-%d')}"

@format_output.register(list)
def _(obj):
    return f"LIST: {', '.join(format_output(x) for x in obj)}"
```

#### 6. @functools.total_ordering - Complete Comparison Operations
 
This decorator generates all comparison methods from a minimal set you define.

Here we create a semantic versioning class with complete comparison capabilities. By defining just \_\_eq\_\_ and \_\_lt\_\_, the @total_ordering decorator automatically generates \_\_le\_\_, \_\_gt\_\_, and \_\_ge\_\_.

```python
from functools import total_ordering

@total_ordering
class Version:
    def __init__(self, major, minor, patch):
        self.major = major
        self.minor = minor
        self.patch = patch

    def __eq__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __lt__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __repr__(self):
        return f"v{self.major}.{self.minor}.{self.patch}"
```

This saves considerable boilerplate code while ensuring consistent behavior across all comparison operations. The Version class can now be fully sorted, compared with all operators, and used in any context that requires ordered objects.

```python
versions = [
    Version(2, 0, 0),
    Version(1, 9, 5),
    Version(1, 11, 0),
    Version(2, 0, 1)
]

print(f"Sorted versions: {sorted(versions)}")
print(f"v1.9.5 > v1.11.0: {Version(1, 9, 5) > Version(1, 11, 0)}")
print(f"v2.0.0 >= v2.0.0: {Version(2, 0, 0) >= Version(2, 0, 0)}")
print(f"v2.0.1 <= v2.0.0: {Version(2, 0, 1) <= Version(2, 0, 0)}")
```

```bash 
# Output:
Sorted versions: [v1.9.5, v1.11.0, v2.0.0, v2.0.1]
v1.9.5 > v1.11.0: False
v2.0.0 >= v2.0.0: True
v2.0.1 <= v2.0.0: False
```


# Descriptors
Descriptors are a low-level mechanism used to manage attribute access in Python. They allow you to customize behavior when attributes are accessed, set, or deleted.
```python
class Descriptor:
    def __get__(self, instance, owner):
        return 'Getting value'

    def __set__(self, instance, value):
        print(f'Setting value: {value}')

class MyClass:
    attr = Descriptor()

obj = MyClass()
print(obj.attr)  # Getting value
obj.attr = 'new value'  # Setting value: new value
```

# Python’s functools Module
* **functools.lru_cache()**: Caches the results of expensive function calls to improve performance.
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def expensive_function(x):
    print(f"Computing {x}")
    return x * 2

print(expensive_function(4))  # Computed
print(expensive_function(4))  # Cached
```

* **functools.partial():** Allows you to fix a certain number of arguments of a function and generate a new function.
```python
from functools import partial

def multiply(x, y):
    return x * y

double = partial(multiply, 2)
print(double(5))  # Output: 10
```

# Context Managers in Python (Using with Statement)
Context managers are used to manage resources like files, database connections, or network connections in a clean and efficient manner. They ensure that resources are properly acquired and released, even if an error occurs.

A context manager is used with the with statement, which makes your code cleaner and less error-prone, especially when working with external resources.

#### Key Concepts of Context Managers
1. **Automatic Setup and Teardown:** A context manager automatically handles resource setup (like opening a file) and teardown (like closing the file), even if an exception occurs.
2. \_\_enter\_\_() and \_\_exit\_\_() Methods: To create a context manager, you need to implement these methods in a class.
    * \_\_enter\_\_(): Called when the with block is entered. It can set up resources and return an object to use inside the block.
    * \_\_exit\_\_(): Called when the with block is exited. It handles cleanup, even if an error occurs within the block.

### Implementing a Context Manager (Custom Class)
Here's an example of how to implement a context manager using a custom class:

```python
class MyContextManager:
    def __enter__(self):
        # Setup code: Open a file, acquire a lock, etc.
        print("Entering the context")
        return self  # This can be any object you want to use inside the with block.

    def __exit__(self, exc_type, exc_value, traceback):
        # Cleanup code: Close the file, release the lock, etc.
        print("Exiting the context")
        if exc_type:
            print(f"An exception occurred: {exc_value}")
        return True  # Return True to suppress the exception, if any.

### Using the context manager with a 'with' statement
with MyContextManager() as cm:
    print("Inside the context")
    # Uncomment the next line to see exception handling
    # raise ValueError("Something went wrong")

print("Outside the context")
```

Output:
```
Entering the context
Inside the context
Exiting the context
Outside the context
```
If an exception were raised inside the with block, the context manager would catch it in the \_\_exit\_\_ method.




# Python Generators
A generator is a special type of iterator in Python that allows you to iterate over a sequence of values **lazily**, one at a time, instead of storing the entire sequence in memory at once. Generators are often used for handling large datasets or streams of data where you want to process one item at a time without loading everything into memory.

### How Generators Work
* **Generators are functions** that use the yield keyword instead of return.
* When a generator function is called, it does not execute the function immediately. Instead, it returns a generator object.
* **Lazy evaluation**: The code inside the generator function runs only when you iterate over it. Each time you call next(), the function resumes execution right after the last yield statement, and it continues until it encounters another yield or the end of the function.
* When a yield is encountered, the value is returned to the caller, and the function's state is saved so it can resume from where it left off the next time next() is called.

### Basic Syntax of a Generator Function
```python
def my_generator():
    yield 1
    yield 2
    yield 3
```

This function will yield values one at a time when iterated over, without generating the entire sequence at once.

### Using a Generator
You can create and use a generator like this:

```python
gen = my_generator()

# You can use next() to get the next value from the generator
print(next(gen))  # Output: 1
print(next(gen))  # Output: 2
print(next(gen))  # Output: 3
```
After all values are yielded, calling next() again will raise the StopIteration exception, indicating that the generator is exhausted.

### Using Generators in a Loop
Generators are commonly used in loops, where you can iterate over each yielded value:

```python
for value in my_generator():
    print(value)
```
This will print:
```
1
2
3
```

### Benefits of Generators
- **Memory Efficiency:**
    - Generators don’t store the entire sequence in memory. They generate values on the fly, which makes them very memory-efficient for large datasets.
    - For example, if you're processing a large file, a generator can yield one line at a time without reading the entire file into memory.
- **Lazy Evaluation:**
    - Values are generated only when needed (on demand). This can lead to performance improvements when working with large data that doesn't need to be fully loaded into memory.
- **Stateful Function:**
    - Generators maintain their state between calls. When the generator is paused at yield, it "remembers" its state (local variables, the next instruction to execute) and resumes from that state the next time it's called.
- **Cleaner Code:**
    - Generators allow you to write clean, readable code without the need to manage complex state or manually keep track of the sequence.

#### Example: Fibonacci Sequence Generator
Here’s an example of a generator that yields the Fibonacci sequence:

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Using the generator to get the first 5 Fibonacci numbers
fib_gen = fibonacci()
for _ in range(5):
    print(next(fib_gen))
```

Output:
```
0
1
1
2
3
```

### Generators vs Iterators
* **Generators**: A generator is a type of iterator created with a function that uses yield. It generates values lazily, meaning that values are only computed when needed.
* **Iterators**: An iterator is any object that implements the __iter__() and __next__() methods. Generators are a specific, more convenient type of iterator.

### Generator Expressions
A generator expression is a more compact way to create generators using a syntax similar to list comprehensions, but instead of using square brackets ([]), they use parentheses (()).

Example of a generator expression:
```python
gen_expr = (x * x for x in range(5))

# Iterating over the generator expression
for value in gen_expr:
    print(value)
```
Output:
```
0
1
4
9
16
```

### Key Differences Between List Comprehensions and Generator Expressions:
- **Memory Efficiency**: List comprehensions create a full list in memory, whereas generator expressions generate one value at a time.
- **Performance**: Generator expressions are more efficient when dealing with large datasets because they don't require creating a whole list in memory.

### Sending Values to a Generator
You can send values to a generator using the send() method. This allows you to provide input to the generator and resume execution from where it last yielded a value.
```python

def count_up_to(max):
    count = 1
    while count <= max:
        received = yield count  # Wait for a value to be sent back
        if received:
            count = received  # If a value is sent, resume counting from that value
        else:
            count += 1

gen = count_up_to(5)

print(next(gen))  # Output: 1
print(next(gen))  # Output: 2
print(gen.send(10))  # Output: 10 (resumes from count = 10)
print(next(gen))  # Output: 11
