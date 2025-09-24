"""
Interquartile Range (IQR)-Based Semantic Chunking
==================================================

This module implements IQR-based semantic chunking using LangChain.
IQR-based chunking uses quartile analysis to identify outliers in similarity 
scores for splitting, making it robust to extreme values.

How it works:
1. Calculate similarity scores between consecutive sentences
2. Find Q1 (25th percentile) and Q3 (75th percentile) of similarity scores
3. Calculate IQR = Q3 - Q1
4. Split where similarity falls below: Q1 - (multiplier × IQR)

When to use:
- When you want robust splitting that ignores extreme values
- For content with occasional very high or very low similarity spikes
- When consistency is more important than sensitivity

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


class IQRSemanticChunker:
    """
    Interquartile Range (IQR)-based semantic chunker implementation.
    """
    
    def __init__(self, embeddings_model=None, iqr_multiplier: float = 1.5):
        """
        Initialize the IQR-based semantic chunker.
        
        Args:
            embeddings_model: Pre-initialized embeddings model (optional)
            iqr_multiplier: IQR multiplier for outlier detection (default: 1.5)
        """
        self.embeddings = embeddings_model or initialize_embeddings()
        self.iqr_multiplier = iqr_multiplier
        
        # Create the semantic chunker
        self.chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="interquartile",
            breakpoint_threshold_amount=self.iqr_multiplier
        )
        
        print(f"🔧 IQR-based chunker initialized with {iqr_multiplier}× IQR multiplier")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Apply IQR-based chunking to the input text.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        print("🔄 Applying IQR-based chunking...")
        
        # Calculate similarity scores for analysis
        sentences, similarity_scores = calculate_similarity_scores(text, self.embeddings)
        
        if similarity_scores:
            q1 = np.percentile(similarity_scores, 25)
            q3 = np.percentile(similarity_scores, 75)
            iqr = q3 - q1
            threshold = q1 - (self.iqr_multiplier * iqr)
            
            print(f"📊 IQR Statistics:")
            print(f"   📉 Q1 (25th percentile): {q1:.3f}")
            print(f"   📈 Q3 (75th percentile): {q3:.3f}")
            print(f"   📊 IQR (Q3 - Q1): {iqr:.3f}")
            print(f"   🎯 Split threshold: {threshold:.3f} (Q1 - {self.iqr_multiplier}×IQR)")
        
        chunks = self.chunker.split_text(text)
        
        display_chunks(chunks, "IQR-Based", "🔺")
        
        return chunks
    
    def adjust_multiplier(self, new_multiplier: float):
        """
        Adjust the IQR multiplier and reinitialize the chunker.
        
        Args:
            new_multiplier: New IQR multiplier
        """
        if new_multiplier < 0:
            raise ValueError("IQR multiplier must be non-negative")
        
        self.iqr_multiplier = new_multiplier
        self.chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="interquartile",
            breakpoint_threshold_amount=self.iqr_multiplier
        )
        
        print(f"⚙️ Multiplier adjusted to {new_multiplier}× IQR")
    
    def analyze_quartile_statistics(self, text: str):
        """
        Analyze the quartile properties of the text's similarity scores.
        
        Args:
            text: Input text to analyze
        """
        sentences, similarity_scores = calculate_similarity_scores(text, self.embeddings)
        
        if not similarity_scores:
            print("❌ Cannot analyze: insufficient sentences in text")
            return
        
        q1 = np.percentile(similarity_scores, 25)
        q2 = np.percentile(similarity_scores, 50)  # Median
        q3 = np.percentile(similarity_scores, 75)
        iqr = q3 - q1
        min_sim = np.min(similarity_scores)
        max_sim = np.max(similarity_scores)
        
        # Identify outliers using standard IQR method
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_low = [score for score in similarity_scores if score < lower_bound]
        outliers_high = [score for score in similarity_scores if score > upper_bound]
        
        print("📊 QUARTILE ANALYSIS")
        print("=" * 50)
        print(f"📉 Q1 (25th percentile): {q1:.3f}")
        print(f"📊 Q2 (Median, 50th percentile): {q2:.3f}")
        print(f"📈 Q3 (75th percentile): {q3:.3f}")
        print(f"📏 IQR (Q3 - Q1): {iqr:.3f}")
        print(f"📉 Minimum: {min_sim:.3f}")
        print(f"📈 Maximum: {max_sim:.3f}")
        
        print(f"\n🎯 OUTLIER DETECTION:")
        print(f"   Lower bound (Q1 - 1.5×IQR): {lower_bound:.3f}")
        print(f"   Upper bound (Q3 + 1.5×IQR): {upper_bound:.3f}")
        print(f"   Low outliers: {len(outliers_low)} scores")
        print(f"   High outliers: {len(outliers_high)} scores")
        
        # Show different threshold options
        print(f"\n🎯 THRESHOLD OPTIONS:")
        for multiplier in [1.0, 1.5, 2.0, 3.0]:
            threshold = q1 - (multiplier * iqr)
            below_threshold = sum(1 for score in similarity_scores if score < threshold)
            print(f"   {multiplier}× IQR: {threshold:.3f} ({below_threshold} splits)")
    
    def compare_with_standard_outliers(self, text: str):
        """
        Compare IQR-based outlier detection with standard statistical methods.
        
        Args:
            text: Input text to analyze
        """
        sentences, similarity_scores = calculate_similarity_scores(text, self.embeddings)
        
        if not similarity_scores:
            print("❌ Cannot analyze: insufficient sentences in text")
            return
        
        # IQR method
        q1 = np.percentile(similarity_scores, 25)
        q3 = np.percentile(similarity_scores, 75)
        iqr = q3 - q1
        iqr_threshold = q1 - 1.5 * iqr
        
        # Standard deviation method
        mean_sim = np.mean(similarity_scores)
        std_sim = np.std(similarity_scores)
        std_threshold = mean_sim - 2 * std_sim  # 2-sigma rule
        
        iqr_outliers = sum(1 for score in similarity_scores if score < iqr_threshold)
        std_outliers = sum(1 for score in similarity_scores if score < std_threshold)
        
        print("🔍 OUTLIER DETECTION COMPARISON")
        print("=" * 50)
        print(f"🔺 IQR Method (Q1 - 1.5×IQR): {iqr_threshold:.3f} → {iqr_outliers} splits")
        print(f"📊 Std Dev Method (μ - 2σ): {std_threshold:.3f} → {std_outliers} splits")
        print(f"\n💡 IQR method is more robust to extreme values!")
    
    def get_info(self):
        """Print information about the IQR-based chunking method."""
        print_method_info(
            method_name="Interquartile Range (IQR)-Based Semantic Chunking",
            description="Uses quartile analysis to identify outliers in similarity scores for robust splitting",
            when_to_use=[
                "Content with occasional similarity spikes or drops",
                "Robust splitting that ignores extreme values",
                "When consistency is more important than sensitivity",
                "Noisy data with outliers that should be ignored"
            ],
            parameters=[
                "Higher multipliers (2.0-3.0) → Very conservative splitting",
                "Standard multiplier (1.5) → Balanced outlier detection",
                "Lower multipliers (1.0) → More sensitive to quartile variations",
                "1.5 is the classical outlier detection threshold"
            ]
        )


def demonstrate_iqr_chunking():
    """
    Demonstrate IQR-based semantic chunking with different multipliers.
    """
    print("🎯 IQR-BASED SEMANTIC CHUNKING DEMONSTRATION")
    print("=" * 80)
    
    # Initialize embeddings once
    embeddings = initialize_embeddings()
    
    # Get sample text
    sample_text = get_sample_text()
    
    # First, analyze the quartile statistics
    print(f"\n🔬 ANALYZING QUARTILE STATISTICS...")
    analyzer = IQRSemanticChunker(embeddings)
    analyzer.analyze_quartile_statistics(sample_text)
    
    # Compare with standard outlier detection
    print(f"\n🔍 COMPARING OUTLIER DETECTION METHODS...")
    analyzer.compare_with_standard_outliers(sample_text)
    
    # Demonstrate different multipliers
    multipliers = [1.0, 1.5, 2.0]
    
    print(f"\n🔬 Testing different IQR multipliers...")
    
    for multiplier in multipliers:
        print(f"\n" + "🔺" * 25 + f" {multiplier}× IQR MULTIPLIER " + "🔺" * 25)
        
        # Create chunker with specific multiplier
        chunker = IQRSemanticChunker(embeddings, multiplier)
        
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
    print(f"   • IQR method is robust to extreme similarity values")
    print(f"   • 1.5 multiplier is the classical outlier detection threshold")
    print(f"   • Higher multipliers are more conservative with splitting")
    print(f"   • Best for content with occasional similarity spikes or noise")


if __name__ == "__main__":
    # Show method information
    chunker = IQRSemanticChunker()
    chunker.get_info()
    
    # Run demonstration
    demonstrate_iqr_chunking()
    
    print("\n✨ IQR-based chunking demonstration complete! ✨")