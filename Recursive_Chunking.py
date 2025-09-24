# Recursive Chunking Example with LangChain
# Install required packages: pip install langchain langchain-text-splitters

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Sample text with different structural elements
sample_text = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence (AI) that focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.

Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.

## Types of Machine Learning

There are several types of machine learning approaches:

### Supervised Learning
Supervised learning uses labeled training data to learn a mapping function from input variables to output variables. The goal is to approximate the mapping function so well that when you have new input data, you can predict the output variables for that data.

Common supervised learning algorithms include:
- Linear regression
- Decision trees
- Random forest
- Support vector machines

### Unsupervised Learning
Unsupervised learning uses machine learning algorithms to analyze and cluster unlabeled datasets. These algorithms discover hidden patterns or data groupings without the need for human intervention.

Popular unsupervised learning techniques include:
- K-means clustering
- Hierarchical clustering
- Principal component analysis (PCA)
- Association rules

### Reinforcement Learning
Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward.

Key concepts in reinforcement learning:
- Agent: The learner or decision maker
- Environment: Everything the agent interacts with
- Action: All possible moves the agent can make
- State: The current situation of the agent
- Reward: Feedback from the environment

## Applications of Machine Learning

Machine learning has numerous applications across various industries:

In healthcare, ML is used for drug discovery, medical imaging analysis, and personalized treatment plans. Financial services use machine learning for fraud detection, algorithmic trading, and credit scoring.

Technology companies leverage ML for recommendation systems, natural language processing, and computer vision applications.

## Conclusion

Machine learning continues to evolve and transform industries. As data becomes more abundant and computing power increases, we can expect even more innovative applications of machine learning in the future.
"""

# Example 1: Basic Recursive Chunking
print("=" * 50)
print("EXAMPLE 1: Basic Recursive Chunking")
print("=" * 50)

# Create a text splitter with default separators
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Small chunk size to see the splitting in action
    chunk_overlap=50,  # Overlap to maintain context
    length_function=len,
)

# Split the text
chunks = text_splitter.split_text(sample_text)

print(f"Number of chunks created: {len(chunks)}")
print()

for i, chunk in enumerate(chunks, 1):
    print(f"--- Chunk {i} (Length: {len(chunk)}) ---")
    print(chunk.strip())
    print()

# Example 2: Custom Separators for Different Document Types
print("=" * 50)
print("EXAMPLE 2: Custom Separators")
print("=" * 50)

# Custom separators prioritizing markdown structure
markdown_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=[
        "\n## ",      # H2 headers
        "\n### ",     # H3 headers
        "\n\n",       # Paragraph breaks
        "\n",         # Line breaks
        " ",          # Spaces
        ""            # Characters
    ]
)

md_chunks = markdown_splitter.split_text(sample_text)
print(f"Number of chunks with custom separators: {len(md_chunks)}")
print()

for i, chunk in enumerate(md_chunks[:3], 1):  # Show first 3 chunks
    print(f"--- Chunk {i} (Length: {len(chunk)}) ---")
    print(chunk.strip())
    print()

# Example 3: Code-Specific Recursive Chunking
print("=" * 50)
print("EXAMPLE 3: Code-Specific Chunking")
print("=" * 50)

python_code = """
def calculate_fibonacci(n):
    '''Calculate fibonacci sequence up to n terms'''
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib_sequence = [0, 1]
    for i in range(2, n):
        next_fib = fib_sequence[i-1] + fib_sequence[i-2]
        fib_sequence.append(next_fib)
    
    return fib_sequence

class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.processed = False
    
    def clean_data(self):
        '''Remove null values and duplicates'''
        cleaned = []
        seen = set()
        for item in self.data:
            if item is not None and item not in seen:
                cleaned.append(item)
                seen.add(item)
        self.data = cleaned
        return self
    
    def transform_data(self, func):
        '''Apply transformation function to all data'''
        if not self.processed:
            self.clean_data()
        self.data = [func(item) for item in self.data]
        self.processed = True
        return self.data

# Usage example
if __name__ == "__main__":
    # Generate fibonacci numbers
    fib_numbers = calculate_fibonacci(10)
    print("Fibonacci sequence:", fib_numbers)
    
    # Process some data
    raw_data = [1, 2, None, 3, 2, 4, None, 5]
    processor = DataProcessor(raw_data)
    result = processor.clean_data().transform_data(lambda x: x * 2)
    print("Processed data:", result)
"""

# Code-specific splitter
code_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=[
        "\nclass ",     # Class definitions
        "\ndef ",       # Function definitions
        "\n\n",         # Double newlines
        "\n",           # Single newlines
        " ",            # Spaces
        ""              # Characters
    ]
)

code_chunks = code_splitter.split_text(python_code)
print(f"Number of code chunks: {len(code_chunks)}")
print()

for i, chunk in enumerate(code_chunks, 1):
    print(f"--- Code Chunk {i} (Length: {len(chunk)}) ---")
    print(chunk.strip())
    print()

# Example 4: Demonstrating the Recursive Nature
print("=" * 50)
print("EXAMPLE 4: Demonstrating Recursive Behavior")
print("=" * 50)

# Very small chunk size to force multiple levels of splitting
small_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Short text to clearly see the recursive splitting
short_text = """This is the first paragraph with multiple sentences. It contains several ideas that should be kept together when possible.

This is the second paragraph. It also has multiple sentences and ideas that form a cohesive unit of thought.

This is the third paragraph with even more content to demonstrate splitting."""

small_chunks = small_splitter.split_text(short_text)

print(f"Number of small chunks: {len(small_chunks)}")
print("Notice how it tries to split on paragraphs first, then sentences, then words:")
print()

for i, chunk in enumerate(small_chunks, 1):
    print(f"Chunk {i} ({len(chunk)} chars): '{chunk.strip()}'")
    print()

# Example 5: Document Processing with Metadata
print("=" * 50)
print("EXAMPLE 5: Document Processing with Metadata")
print("=" * 50)

from langchain.docstore.document import Document

# Create documents from chunks with metadata
documents = []
for i, chunk in enumerate(chunks[:3]):  # Use first 3 chunks from example 1
    doc = Document(
        page_content=chunk,
        metadata={
            "chunk_id": i,
            "chunk_size": len(chunk),
            "source": "machine_learning_guide",
            "splitting_method": "recursive"
        }
    )
    documents.append(doc)

print("Documents with metadata:")
for doc in documents:
    print(f"Chunk ID: {doc.metadata['chunk_id']}")
    print(f"Size: {doc.metadata['chunk_size']} characters")
    print(f"Content preview: {doc.page_content[:100]}...")
    print("-" * 30)