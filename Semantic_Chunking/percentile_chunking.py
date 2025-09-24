"""
Percentile-Based Semantic Chunking
===================================

This module implements percentile-based semantic chunking using LangChain.
Percentile-based chunking splits text at points where the similarity between 
consecutive sentences falls below a specified percentile threshold.

How it works:
1. Calculate semantic similarity between all consecutive sentence pairs
2. Determine the percentile threshold (e.g., 75th percentile)
3. Split the text wherever similarity drops below this threshold

When to use:
- When you want chunks of relatively consistent quality
- For balanced chunk sizes across different content types
- When you need predictable chunking behavior

"""

from langchain_experimental.text_splitter import SemanticChunker
from typing import List
from utils import (
    initialize_embeddings, 
    get_sample_text, 
    display_chunks, 
    print_method_info
)


class PercentileSemanticChunker:
    """
    Percentile-based semantic chunker implementation.
    """
    
    def __init__(self, embeddings_model=None, percentile_threshold: float = 75):
        """
        Initialize the percentile-based semantic chunker.
        
        Args:
            embeddings_model: Pre-initialized embeddings model (optional)
            percentile_threshold: Percentile threshold for splitting (default: 75)
        """
        self.embeddings = embeddings_model or initialize_embeddings()
        self.percentile_threshold = percentile_threshold
        
        # Create the semantic chunker
        self.chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=self.percentile_threshold
        )
        
        print(f"🔧 Percentile-based chunker initialized with {percentile_threshold}th percentile threshold")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Apply percentile-based chunking to the input text.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        print("🔄 Applying percentile-based chunking...")
        chunks = self.chunker.split_text(text)
        
        display_chunks(chunks, "Percentile-Based", "🔹")
        
        return chunks
    
    def adjust_threshold(self, new_threshold: float):
        """
        Adjust the percentile threshold and reinitialize the chunker.
        
        Args:
            new_threshold: New percentile threshold (0-100)
        """
        if not 0 <= new_threshold <= 100:
            raise ValueError("Percentile threshold must be between 0 and 100")
        
        self.percentile_threshold = new_threshold
        self.chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=self.percentile_threshold
        )
        
        print(f"⚙️ Threshold adjusted to {new_threshold}th percentile")
    
    def get_info(self):
        """Print information about the percentile-based chunking method."""
        print_method_info(
            method_name="Percentile-Based Semantic Chunking",
            description="Splits text at points where similarity drops below a specified percentile threshold",
            when_to_use=[
                "Consistent chunk quality across different content types",
                "Balanced chunk sizes for most general use cases", 
                "Predictable chunking behavior needed",
                "Starting point for semantic chunking experiments"
            ],
            parameters=[
                "Higher percentiles (80-95) → Fewer, larger chunks",
                "Lower percentiles (60-75) → More, smaller chunks",
                "75th percentile is typically a good starting point",
                "Adjust based on your specific content and use case"
            ]
        )


def demonstrate_percentile_chunking():
    """
    Demonstrate percentile-based semantic chunking with different thresholds.
    """
    print("🎯 PERCENTILE-BASED SEMANTIC CHUNKING DEMONSTRATION")
    print("=" * 80)
    
    # Initialize embeddings once
    embeddings = initialize_embeddings()
    
    # Get sample text
    sample_text = get_sample_text()
    
    # Demonstrate different percentile thresholds
    thresholds = [60, 75, 90]
    
    print(f"\n🔬 Testing different percentile thresholds...")
    
    for threshold in thresholds:
        print(f"\n" + "🔸" * 30 + f" {threshold}th PERCENTILE " + "🔸" * 30)
        
        # Create chunker with specific threshold
        chunker = PercentileSemanticChunker(embeddings, threshold)
        
        # Apply chunking
        chunks = chunker.chunk_text(sample_text)
        
        # Summary statistics
        chunk_lengths = [len(chunk) for chunk in chunks]
        print(f"\n📊 STATISTICS FOR {threshold}th PERCENTILE:")
        print(f"   📈 Total chunks: {len(chunks)}")
        print(f"   📏 Average length: {sum(chunk_lengths) / len(chunk_lengths):.0f} characters")
        print(f"   📐 Min length: {min(chunk_lengths)} characters")
        print(f"   📐 Max length: {max(chunk_lengths)} characters")
    
    print(f"\n💡 INSIGHTS:")
    print(f"   • Higher percentiles create fewer, larger chunks")
    print(f"   • Lower percentiles create more, smaller chunks")
    print(f"   • 75th percentile typically provides good balance")
    print(f"   • Adjust based on your downstream application needs")


if __name__ == "__main__":
    # Show method information
    chunker = PercentileSemanticChunker()
    chunker.get_info()
    
    # Run demonstration
    demonstrate_percentile_chunking()
    
    print("\n✨ Percentile-based chunking demonstration complete! ✨")