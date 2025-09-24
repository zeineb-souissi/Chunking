"""
Semantic Chunking Comparison and Analysis
==========================================

This module provides comprehensive comparison and visualization tools for 
analyzing different semantic chunking methods. It includes statistical 
comparisons, visual plots, and detailed analysis of chunking behavior.

Features:
- Side-by-side comparison of all three chunking methods
- Statistical analysis and visualization
- Similarity score analysis with threshold visualization
- Performance metrics and recommendations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Tuple
from utils import (
    initialize_embeddings, 
    get_sample_text, 
    calculate_similarity_scores
)
from percentile_chunking import PercentileSemanticChunker
from std_dev_chunking import StandardDeviationSemanticChunker
from iqr_chunking import IQRSemanticChunker


class SemanticChunkingComparator:
    """
    Comprehensive comparison tool for semantic chunking methods.
    """
    
    def __init__(self, embeddings_model=None):
        """
        Initialize the comparator with an embeddings model.
        
        Args:
            embeddings_model: Pre-initialized embeddings model (optional)
        """
        self.embeddings = embeddings_model or initialize_embeddings()
        
        # Initialize all chunking methods with default parameters
        self.percentile_chunker = PercentileSemanticChunker(self.embeddings, 75)
        self.std_dev_chunker = StandardDeviationSemanticChunker(self.embeddings, 1.0)
        self.iqr_chunker = IQRSemanticChunker(self.embeddings, 1.5)
        
        print("🔧 Semantic chunking comparator initialized with all three methods")
    
    def compare_all_methods(self, text: str) -> Dict[str, List[str]]:
        """
        Apply all three chunking methods to the same text and return results.
        
        Args:
            text: Input text to chunk
            
        Returns:
            Dictionary mapping method names to their chunks
        """
        print("🔄 Applying all three chunking methods...")
        
        # Apply each method
        percentile_chunks = self.percentile_chunker.chunk_text(text)
        std_dev_chunks = self.std_dev_chunker.chunk_text(text)
        iqr_chunks = self.iqr_chunker.chunk_text(text)
        
        return {
            'Percentile-Based': percentile_chunks,
            'Standard Deviation': std_dev_chunks,
            'IQR-Based': iqr_chunks
        }
    
    def create_comparison_table(self, chunks_dict: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Create a detailed comparison table of chunking results.
        
        Args:
            chunks_dict: Dictionary mapping method names to chunks
            
        Returns:
            Pandas DataFrame with comparison statistics
        """
        comparison_data = {
            'Method': [],
            'Number of Chunks': [],
            'Avg Chunk Length': [],
            'Min Chunk Length': [],
            'Max Chunk Length': [],
            'Std Dev Length': []
        }
        
        for method_name, chunks in chunks_dict.items():
            chunk_lengths = [len(chunk) for chunk in chunks]
            
            comparison_data['Method'].append(method_name)
            comparison_data['Number of Chunks'].append(len(chunks))
            comparison_data['Avg Chunk Length'].append(int(np.mean(chunk_lengths)))
            comparison_data['Min Chunk Length'].append(min(chunk_lengths))
            comparison_data['Max Chunk Length'].append(max(chunk_lengths))
            comparison_data['Std Dev Length'].append(int(np.std(chunk_lengths)))
        
        return pd.DataFrame(comparison_data)
    
    def visualize_comparisons(self, chunks_dict: Dict[str, List[str]]):
        """
        Create comprehensive visualizations comparing all methods.
        
        Args:
            chunks_dict: Dictionary mapping method names to chunks
        """
        # Create comparison DataFrame
        comparison_df = self.create_comparison_table(chunks_dict)
        
        print("📊 CHUNKING METHODS COMPARISON")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        
        # Create visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        methods = comparison_df['Method']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # 1. Number of chunks comparison
        chunk_counts = comparison_df['Number of Chunks']
        bars1 = ax1.bar(methods, chunk_counts, color=colors, alpha=0.8)
        ax1.set_title('Number of Chunks Created', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Chunks')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars1, chunk_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                     f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Average chunk length comparison
        bars2 = ax2.bar(methods, comparison_df['Avg Chunk Length'], color=colors, alpha=0.8)
        ax2.set_title('Average Chunk Length', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Characters')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, length in zip(bars2, comparison_df['Avg Chunk Length']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                     f'{length}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Chunk length distribution (box plot)
        chunk_lengths_data = {
            'Percentile': [len(chunk) for chunk in chunks_dict['Percentile-Based']],
            'Std Dev': [len(chunk) for chunk in chunks_dict['Standard Deviation']],
            'IQR': [len(chunk) for chunk in chunks_dict['IQR-Based']]
        }
        
        box_data = [chunk_lengths_data[method] for method in ['Percentile', 'Std Dev', 'IQR']]
        box_plot = ax3.boxplot(box_data, labels=['Percentile', 'Std Dev', 'IQR'], 
                               patch_artist=True, notch=True)
        
        # Color the box plots
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_title('Chunk Length Distribution', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Characters')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Min vs Max chunk lengths
        x_pos = np.arange(len(methods))
        width = 0.35
        
        bars_min = ax4.bar(x_pos - width/2, comparison_df['Min Chunk Length'], 
                           width, label='Min Length', color='lightcoral', alpha=0.8)
        bars_max = ax4.bar(x_pos + width/2, comparison_df['Max Chunk Length'], 
                           width, label='Max Length', color='lightblue', alpha=0.8)
        
        ax4.set_title('Min vs Max Chunk Lengths', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Characters')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(methods)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars_min:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
                     f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars_max:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
                     f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        print("\n🎯 KEY INSIGHTS:")
        print("• Different methods create different numbers of chunks")
        print("• Chunk size consistency varies between methods")
        print("• Each method responds differently to content transitions")
    
    def visualize_similarity_thresholds(self, text: str):
        """
        Visualize similarity scores and thresholds for all methods.
        
        Args:
            text: Input text to analyze
        """
        sentences, similarity_scores = calculate_similarity_scores(text, self.embeddings)
        
        if not similarity_scores:
            print("❌ Cannot visualize: insufficient sentences in text")
            return
        
        # Calculate thresholds for each method
        mean_sim = np.mean(similarity_scores)
        std_sim = np.std(similarity_scores)
        q1 = np.percentile(similarity_scores, 25)
        q3 = np.percentile(similarity_scores, 75)
        iqr = q3 - q1
        percentile_75 = np.percentile(similarity_scores, 75)
        
        # Method thresholds
        std_threshold = mean_sim - 1.0 * std_sim
        iqr_threshold = q1 - 1.5 * iqr
        percentile_threshold = percentile_75
        
        print(f"📊 Calculated {len(similarity_scores)} similarity scores between consecutive sentences")
        print(f"📈 Similarity range: {min(similarity_scores):.3f} to {max(similarity_scores):.3f}")
        print(f"📉 Mean similarity: {mean_sim:.3f}")
        print(f"📊 Standard deviation: {std_sim:.3f}")
        
        # Create detailed similarity visualization
        plt.figure(figsize=(15, 10))
        
        # Plot similarity scores
        plt.subplot(2, 1, 1)
        sentence_positions = range(1, len(similarity_scores) + 1)
        plt.plot(sentence_positions, similarity_scores, 'b-', linewidth=2, alpha=0.7, label='Similarity Scores')
        plt.fill_between(sentence_positions, similarity_scores, alpha=0.3)
        
        # Add threshold lines
        plt.axhline(y=percentile_threshold, color='#FF6B6B', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'Percentile (75th): {percentile_threshold:.3f}')
        plt.axhline(y=std_threshold, color='#4ECDC4', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'Std Dev (μ-1σ): {std_threshold:.3f}')
        plt.axhline(y=iqr_threshold, color='#45B7D1', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'IQR (Q1-1.5×IQR): {iqr_threshold:.3f}')
        
        plt.title('Semantic Similarity Scores and Method Thresholds', fontsize=14, fontweight='bold')
        plt.xlabel('Sentence Position')
        plt.ylabel('Cosine Similarity')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Create histogram of similarity scores
        plt.subplot(2, 1, 2)
        plt.hist(similarity_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=percentile_threshold, color='#FF6B6B', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'Percentile: {percentile_threshold:.3f}')
        plt.axvline(x=std_threshold, color='#4ECDC4', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'Std Dev: {std_threshold:.3f}')
        plt.axvline(x=iqr_threshold, color='#45B7D1', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'IQR: {iqr_threshold:.3f}')
        
        plt.title('Distribution of Similarity Scores with Thresholds', fontsize=14, fontweight='bold')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\n🔍 THRESHOLD ANALYSIS:")
        print(f"   🔴 Percentile method splits when similarity < {percentile_threshold:.3f}")
        print(f"   🟢 Standard Deviation method splits when similarity < {std_threshold:.3f}")
        print(f"   🔵 IQR method splits when similarity < {iqr_threshold:.3f}")
        print("\n💡 The visualization shows WHERE each method would split the text!")
    
    def generate_recommendations(self, chunks_dict: Dict[str, List[str]]) -> str:
        """
        Generate recommendations based on chunking analysis.
        
        Args:
            chunks_dict: Dictionary mapping method names to chunks
            
        Returns:
            String with recommendations
        """
        comparison_df = self.create_comparison_table(chunks_dict)
        
        recommendations = []
        
        # Analyze chunk count consistency
        chunk_counts = comparison_df['Number of Chunks'].values
        if max(chunk_counts) - min(chunk_counts) <= 2:
            recommendations.append("All methods produce similar chunk counts - choose based on other criteria")
        else:
            most_chunks_method = comparison_df.loc[comparison_df['Number of Chunks'].idxmax(), 'Method']
            least_chunks_method = comparison_df.loc[comparison_df['Number of Chunks'].idxmin(), 'Method']
            recommendations.append(f"Large variation in chunk count: {most_chunks_method} creates most chunks, {least_chunks_method} creates fewest")
        
        # Analyze chunk size consistency
        std_devs = comparison_df['Std Dev Length'].values
        most_consistent_idx = np.argmin(std_devs)
        most_consistent_method = comparison_df.loc[most_consistent_idx, 'Method']
        recommendations.append(f"Most consistent chunk sizes: {most_consistent_method}")
        
        # General recommendations
        recommendations.extend([
            "For general use: Start with Percentile-Based (75th percentile)",
            "For varying content density: Try Standard Deviation-Based",
            "For robust outlier handling: Use IQR-Based",
            "Always test with your specific content and downstream application"
        ])
        
        return "\n".join(f"• {rec}" for rec in recommendations)


def run_comprehensive_analysis():
    """
    Run a comprehensive analysis comparing all semantic chunking methods.
    """
    print("🎯 COMPREHENSIVE SEMANTIC CHUNKING ANALYSIS")
    print("=" * 80)
    
    # Initialize comparator
    comparator = SemanticChunkingComparator()
    
    # Get sample text
    sample_text = get_sample_text()
    
    # Compare all methods
    print(f"\n🔬 APPLYING ALL THREE METHODS...")
    chunks_dict = comparator.compare_all_methods(sample_text)
    
    # Create visualizations
    print(f"\n📊 CREATING COMPARISON VISUALIZATIONS...")
    comparator.visualize_comparisons(chunks_dict)
    
    # Visualize similarity thresholds
    print(f"\n🔍 ANALYZING SIMILARITY SCORES AND THRESHOLDS...")
    comparator.visualize_similarity_thresholds(sample_text)
    
    # Generate recommendations
    print(f"\n💡 GENERATING RECOMMENDATIONS...")
    recommendations = comparator.generate_recommendations(chunks_dict)
    
    print("\n🎯 RECOMMENDATIONS:")
    print("=" * 50)
    print(recommendations)
    
    # Final summary matrix
    final_summary = {
        'Aspect': [
            'Predictability',
            'Adaptability', 
            'Outlier Resistance',
            'Parameter Sensitivity',
            'Computational Cost',
            'Best Use Case'
        ],
        'Percentile': [
            'High',
            'Medium',
            'Medium', 
            'Low',
            'Low',
            'General purpose'
        ],
        'Standard Deviation': [
            'Medium',
            'High',
            'Low',
            'Medium', 
            'Low',
            'Variable content'
        ],
        'IQR': [
            'High',
            'Medium',
            'High',
            'Low',
            'Low', 
            'Noisy data'
        ]
    }
    
    summary_df = pd.DataFrame(final_summary)
    print("\n📊 METHOD COMPARISON MATRIX:")
    print("=" * 60)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_comprehensive_analysis()
    print(f"\n✨ Comprehensive semantic chunking analysis complete! ✨")