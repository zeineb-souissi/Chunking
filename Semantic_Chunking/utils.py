"""
Semantic Chunking Utilities
============================

This module contains shared utilities for semantic chunking demonstrations,
including embedding model initialization, sample text, and helper functions.

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Tuple
import warnings

# LangChain imports for semantic chunking
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


def initialize_embeddings():
    """
    Initialize the embedding model for semantic analysis.
    
    Returns:
        HuggingFaceEmbeddings: Configured embedding model
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # Fast and efficient for semantic similarity
        model_kwargs={'device': 'cpu'}  # Use CPU for compatibility
    )
    
    print("🔧 Embedding model initialized successfully!")
    print("📊 Model: all-MiniLM-L6-v2 (384-dimensional embeddings)")
    print("⚡ Optimized for semantic similarity tasks")
    
    return embeddings


def get_sample_text():
    """
    Get the sample text for demonstration purposes.
    
    Returns:
        str: Multi-topic sample text for chunking demonstration
    """
    sample_text = """
Artificial intelligence has revolutionized many aspects of modern life. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions. Deep learning, a subset of machine learning, uses neural networks with multiple layers to solve complex problems.

The applications of AI are diverse and growing rapidly. In healthcare, AI systems can analyze medical images to detect diseases early. Computer vision algorithms can identify tumors in X-rays with accuracy comparable to experienced radiologists. Natural language processing enables chatbots to understand and respond to patient queries.

Climate change represents one of the most pressing challenges of our time. Global temperatures have risen significantly over the past century due to increased greenhouse gas emissions. The burning of fossil fuels releases carbon dioxide into the atmosphere, creating a greenhouse effect that traps heat.

Renewable energy sources offer hope for reducing our carbon footprint. Solar panels convert sunlight directly into electricity through photovoltaic cells. Wind turbines harness the kinetic energy of moving air to generate power. Hydroelectric dams use flowing water to produce clean energy.

The ocean covers more than 70% of Earth's surface and plays a crucial role in regulating climate. Ocean currents distribute heat around the globe, affecting weather patterns worldwide. The deep sea remains largely unexplored, with scientists estimating that we have mapped less than 20% of the ocean floor.

Marine biodiversity is incredibly rich, with countless species adapted to various oceanic environments. Coral reefs support about 25% of all marine species despite covering less than 1% of the ocean area. These ecosystems are particularly vulnerable to rising sea temperatures and ocean acidification.

Space exploration has captured human imagination for decades. The Apollo missions successfully landed humans on the Moon in the 1960s and 1970s. Recent missions to Mars have provided valuable insights into the planet's geology and potential for past or present life.

Private companies are now playing an increasingly important role in space exploration. SpaceX has developed reusable rockets that significantly reduce launch costs. Blue Origin and Virgin Galactic are working on space tourism initiatives that could make space travel accessible to civilians.

Quantum computing represents a paradigm shift in computational capabilities. Unlike classical computers that use bits (0s and 1s), quantum computers use quantum bits or qubits that can exist in multiple states simultaneously. This quantum superposition allows quantum computers to perform certain calculations exponentially faster than classical computers.
"""
    
    print("📝 Sample text prepared!")
    print(f"📊 Text length: {len(sample_text)} characters")
    print(f"📄 Approximate sentences: {sample_text.count('.')} sentences")
    print("\n🔍 This text covers multiple topics:")
    print("   • Artificial Intelligence & Machine Learning")
    print("   • Climate Change & Renewable Energy") 
    print("   • Ocean Science & Marine Biology")
    print("   • Space Exploration & Private Space Companies")
    print("   • Quantum Computing")
    print("\n💡 Each topic transition will help us see how different chunking methods work!")
    
    return sample_text


def display_chunks(chunks: List[str], method_name: str, boundary_symbol: str = "🔹"):
    """
    Display chunks with clear formatting and boundaries.
    
    Args:
        chunks: List of text chunks
        method_name: Name of the chunking method
        boundary_symbol: Symbol to use for chunk boundaries
    """
    print(f"\n✅ {method_name} chunking complete!")
    print(f"📊 Number of chunks created: {len(chunks)}")
    print(f"📏 Average chunk length: {sum(len(chunk) for chunk in chunks) / len(chunks):.0f} characters")

    # Display the chunks with clear separation
    print("\n" + "="*80)
    print(f"{method_name.upper()} CHUNKS")
    print("="*80)

    for i, chunk in enumerate(chunks, 1):
        print(f"\n📄 CHUNK {i}:")
        print(f"📐 Length: {len(chunk)} characters")
        print("-" * 50)
        print(chunk.strip())
        if i < len(chunks):
            print("\n" + boundary_symbol * 20 + " CHUNK BOUNDARY " + boundary_symbol * 20)


def calculate_similarity_scores(text: str, embeddings_model) -> Tuple[List[str], List[float]]:
    """
    Calculate similarity scores between consecutive sentences.
    
    Args:
        text: Input text to analyze
        embeddings_model: Embedding model to use
        
    Returns:
        Tuple of (sentences, similarity_scores)
    """
    import re
    
    # Split text into sentences (simple approach)
    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    if len(sentences) < 2:
        return [], []
    
    # Calculate embeddings for all sentences
    sentence_embeddings = embeddings_model.embed_documents(sentences)
    similarity_scores = []
    
    # Calculate cosine similarity between consecutive sentences
    for i in range(len(sentence_embeddings) - 1):
        # Calculate cosine similarity
        dot_product = np.dot(sentence_embeddings[i], sentence_embeddings[i + 1])
        norm1 = np.linalg.norm(sentence_embeddings[i])
        norm2 = np.linalg.norm(sentence_embeddings[i + 1])
        similarity = dot_product / (norm1 * norm2)
        similarity_scores.append(similarity)
    
    return sentences, similarity_scores


def print_method_info(method_name: str, description: str, when_to_use: List[str], parameters: List[str]):
    """
    Print formatted information about a chunking method.
    
    Args:
        method_name: Name of the method
        description: How the method works
        when_to_use: List of use cases
        parameters: List of parameter adjustment tips
    """
    print("="*80)
    print(f"{method_name.upper()}")
    print("="*80)
    print(f"\n📋 HOW IT WORKS:")
    print(f"   {description}")
    
    print(f"\n🎯 WHEN TO USE:")
    for use_case in when_to_use:
        print(f"   • {use_case}")
    
    print(f"\n⚙️ PARAMETER TIPS:")
    for param in parameters:
        print(f"   • {param}")
    print()


if __name__ == "__main__":
    print("🔧 Semantic Chunking Utils - Testing")
    print("="*50)
    
    # Test embedding initialization
    embeddings = initialize_embeddings()
    
    # Test sample text
    text = get_sample_text()
    
    # Test similarity calculation
    sentences, scores = calculate_similarity_scores(text, embeddings)
    print(f"\n📊 Calculated {len(scores)} similarity scores")
    print(f"📈 Similarity range: {min(scores):.3f} to {max(scores):.3f}")
    
    print("\n✅ All utilities working correctly!")