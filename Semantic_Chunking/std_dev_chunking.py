"""
Standard Deviation-Based Semantic Chunking
===========================================

This module implements standard deviation-based semantic chunking using LangChain.
Standard deviation-based chunking uses statistical analysis to identify significant 
drops in similarity scores between consecutive sentences.

How it works:
1. Calculate similarity scores between consecutive sentences
2. Compute the mean and standard deviation of these scores
3. Split where similarity falls below: mean - (multiplier × standard_deviation)

When to use:
- When content has varying semantic density
- For adaptive chunking based on content complexity
- When you want statistically-driven split points

"""

from langchain_experimental.text_splitter import SemanticChunker
from typing import List
from utils import (
    initialize_embeddings, 
    get_sample_text, 
    display_chunks, 
    print_method_info,
    calculate_similarity_scores
)
import numpy as np


class StandardDeviationSemanticChunker:
    """
    Standard deviation-based semantic chunker implementation.
    """
    
    def __init__(self, embeddings_model=None, std_multiplier: float = 1.0):
        """
        Initialize the standard deviation-based semantic chunker.
        
        Args:
            embeddings_model: Pre-initialized embeddings model (optional)
            std_multiplier: Standard deviation multiplier for threshold (default: 1.0)
        """
        self.embeddings = embeddings_model or initialize_embeddings()
        self.std_multiplier = std_multiplier
        
        # Create the semantic chunker
        self.chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=self.std_multiplier
        )
        
        print(f"🔧 Standard deviation-based chunker initialized with {std_multiplier}× std dev multiplier")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Apply standard deviation-based chunking to the input text.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        print("🔄 Applying standard deviation-based chunking...")
        
        # Calculate similarity scores for analysis
        sentences, similarity_scores = calculate_similarity_scores(text, self.embeddings)
        
        if similarity_scores:
            mean_sim = np.mean(similarity_scores)
            std_sim = np.std(similarity_scores)
            threshold = mean_sim - (self.std_multiplier * std_sim)
            
            print(f"📊 Similarity Statistics:")
            print(f"   📈 Mean similarity: {mean_sim:.3f}")
            print(f"   📊 Standard deviation: {std_sim:.3f}")
            print(f"   🎯 Split threshold: {threshold:.3f} (mean - {self.std_multiplier}×std)")
        
        chunks = self.chunker.split_text(text)
        
        display_chunks(chunks, "Standard Deviation-Based", "🔸")
        
        return chunks
    
    def adjust_multiplier(self, new_multiplier: float):
        """
        Adjust the standard deviation multiplier and reinitialize the chunker.
        
        Args:
            new_multiplier: New standard deviation multiplier
        """
        if new_multiplier < 0:
            raise ValueError("Standard deviation multiplier must be non-negative")
        
        self.std_multiplier = new_multiplier
        self.chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=self.std_multiplier
        )
        
        print(f"⚙️ Multiplier adjusted to {new_multiplier}× standard deviation")
    
    def analyze_text_statistics(self, text: str):
        """
        Analyze the statistical properties of the text's similarity scores.
        
        Args:
            text: Input text to analyze
        """
        sentences, similarity_scores = calculate_similarity_scores(text, self.embeddings)
        
        if not similarity_scores:
            print("❌ Cannot analyze: insufficient sentences in text")
            return
        
        mean_sim = np.mean(similarity_scores)
        std_sim = np.std(similarity_scores)
        min_sim = np.min(similarity_scores)
        max_sim = np.max(similarity_scores)
        
        print("📊 TEXT SIMILARITY STATISTICS")
        print("=" * 50)
        print(f"📈 Mean similarity: {mean_sim:.3f}")
        print(f"📊 Standard deviation: {std_sim:.3f}")
        print(f"📉 Minimum similarity: {min_sim:.3f}")
        print(f"📈 Maximum similarity: {max_sim:.3f}")
        print(f"📏 Range: {max_sim - min_sim:.3f}")
        
        # Show different threshold options
        print(f"\n🎯 THRESHOLD OPTIONS:")
        for multiplier in [0.5, 1.0, 1.5, 2.0]:
            threshold = mean_sim - (multiplier * std_sim)
            below_threshold = sum(1 for score in similarity_scores if score < threshold)
            print(f"   {multiplier}× std dev: {threshold:.3f} ({below_threshold} splits)")
    
    def get_info(self):
        """Print information about the standard deviation-based chunking method."""
        print_method_info(
            method_name="Standard Deviation-Based Semantic Chunking",
            description="Uses statistical analysis to identify significant drops in similarity scores",
            when_to_use=[
                "Content with varying semantic density",
                "Adaptive chunking based on content complexity",
                "Statistically-driven split points needed",
                "When you want sensitivity to content variations"
            ],
            parameters=[
                "Higher multipliers (1.5-2.0) → More conservative splitting (fewer chunks)",
                "Lower multipliers (0.5-1.0) → More aggressive splitting (more chunks)",
                "1.0 multiplier is typically a good starting point",
                "Analyze text statistics first to choose optimal multiplier"
            ]
        )


def demonstrate_std_dev_chunking():
    """
    Demonstrate standard deviation-based semantic chunking with different multipliers.
    """
    print("🎯 STANDARD DEVIATION-BASED SEMANTIC CHUNKING DEMONSTRATION")
    print("=" * 80)
    
    # Initialize embeddings once
    embeddings = initialize_embeddings()
    
    # Get sample text
    sample_text = get_sample_text()
    
    # First, analyze the text statistics
    print(f"\n🔬 ANALYZING TEXT STATISTICS...")
    analyzer = StandardDeviationSemanticChunker(embeddings)
    analyzer.analyze_text_statistics(sample_text)
    
    # Demonstrate different multipliers
    multipliers = [0.5, 1.0, 1.5]
    
    print(f"\n🔬 Testing different standard deviation multipliers...")
    
    for multiplier in multipliers:
        print(f"\n" + "🔸" * 25 + f" {multiplier}× STD DEV MULTIPLIER " + "🔸" * 25)
        
        # Create chunker with specific multiplier
        chunker = StandardDeviationSemanticChunker(embeddings, multiplier)
        
        # Apply chunking
        chunks = chunker.chunk_text(sample_text)
        
        # Summary statistics
        chunk_lengths = [len(chunk) for chunk in chunks]
        print(f"\n📊 STATISTICS FOR {multiplier}× MULTIPLIER:")
        print(f"   📈 Total chunks: {len(chunks)}")
        print(f"   📏 Average length: {sum(chunk_lengths) / len(chunk_lengths):.0f} characters")
        print(f"   📐 Min length: {min(chunk_lengths)} characters")
        print(f"   📐 Max length: {max(chunk_lengths)} characters")
    
    print(f"\n💡 INSIGHTS:")
    print(f"   • Lower multipliers are more sensitive to similarity variations")
    print(f"   • Higher multipliers create fewer, more conservative splits")
    print(f"   • This method adapts to the statistical properties of your text")
    print(f"   • Best for content with varying semantic density")


if __name__ == "__main__":
    # Show method information
    chunker = StandardDeviationSemanticChunker()
    chunker.get_info()
    
    # Run demonstration
    demonstrate_std_dev_chunking()
    
    print("\n✨ Standard deviation-based chunking demonstration complete! ✨")