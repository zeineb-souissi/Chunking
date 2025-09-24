"""
Semantic Chunking Demonstration - Main Entry Point
===================================================

This is the main demonstration script that showcases all three semantic chunking
methods available in LangChain. It provides an interactive menu system to explore
each method individually or run comprehensive comparisons.

Features:
- Interactive menu system
- Individual method demonstrations
- Comprehensive comparison analysis
- Parameter experimentation
- Visual analysis and recommendations

Run this script to explore semantic chunking capabilities!

Usage:
    python main.py

"""

import sys
from typing import Optional
from utils import initialize_embeddings, get_sample_text
from percentile_chunking import PercentileSemanticChunker, demonstrate_percentile_chunking
from std_dev_chunking import StandardDeviationSemanticChunker, demonstrate_std_dev_chunking
from iqr_chunking import IQRSemanticChunker, demonstrate_iqr_chunking
from comparison_analysis import run_comprehensive_analysis


def print_welcome_banner():
    """Print the welcome banner and introduction."""
    print("🎉" * 30)
    print("   SEMANTIC CHUNKING IN LANGCHAIN")
    print("     Complete Demonstration Suite")
    print("🎉" * 30)
    print()
    print("Welcome to the comprehensive semantic chunking demonstration!")
    print("This suite includes all three LangChain semantic chunking methods:")
    print()
    print("📊 1. Percentile-Based Chunking")
    print("   → Splits at similarity drops below percentile threshold")
    print("   → Best for: Consistent, predictable chunks")
    print()
    print("📈 2. Standard Deviation-Based Chunking") 
    print("   → Uses statistical analysis for adaptive splitting")
    print("   → Best for: Variable content density")
    print()
    print("📉 3. IQR-Based Chunking")
    print("   → Robust outlier detection for stable splitting")
    print("   → Best for: Noisy data with outliers")
    print()
    print("🔍 4. Comprehensive Analysis")
    print("   → Compare all methods with visualizations")
    print("   → Get recommendations for your use case")
    print()


def print_main_menu():
    """Print the main menu options."""
    print("🔧 MAIN MENU")
    print("=" * 50)
    print("1. 📊 Percentile-Based Chunking Demo")
    print("2. 📈 Standard Deviation-Based Chunking Demo")
    print("3. 📉 IQR-Based Chunking Demo")
    print("4. 🔍 Comprehensive Analysis & Comparison")
    print("5. 🎯 Quick Method Info")
    print("6. 🧪 Parameter Experimentation")
    print("7. 📝 Use Your Own Text")
    print("8. ❓ Help & Documentation")
    print("0. 🚪 Exit")
    print("=" * 50)


def show_method_info():
    """Show detailed information about all methods."""
    print("📚 SEMANTIC CHUNKING METHODS OVERVIEW")
    print("=" * 60)
    
    # Percentile method info
    percentile_chunker = PercentileSemanticChunker()
    percentile_chunker.get_info()
    
    # Standard deviation method info
    std_dev_chunker = StandardDeviationSemanticChunker()
    std_dev_chunker.get_info()
    
    # IQR method info
    iqr_chunker = IQRSemanticChunker()
    iqr_chunker.get_info()


def parameter_experimentation():
    """Interactive parameter experimentation."""
    print("🧪 PARAMETER EXPERIMENTATION")
    print("=" * 50)
    print("Choose a method to experiment with parameters:")
    print("1. Percentile-Based (adjust percentile threshold)")
    print("2. Standard Deviation (adjust multiplier)")
    print("3. IQR-Based (adjust IQR multiplier)")
    print("0. Back to main menu")
    
    choice = input("\nEnter your choice (0-3): ").strip()
    
    if choice == "0":
        return
    
    sample_text = get_sample_text()
    embeddings = initialize_embeddings()
    
    if choice == "1":
        print("\n🔬 PERCENTILE THRESHOLD EXPERIMENTATION")
        print("Testing different percentile thresholds...")
        
        for threshold in [60, 70, 75, 80, 85, 90]:
            print(f"\n{'='*20} {threshold}th PERCENTILE {'='*20}")
            chunker = PercentileSemanticChunker(embeddings, threshold)
            chunks = chunker.chunk_text(sample_text)
            print(f"Result: {len(chunks)} chunks created")
    
    elif choice == "2":
        print("\n🔬 STANDARD DEVIATION MULTIPLIER EXPERIMENTATION")
        print("Testing different standard deviation multipliers...")
        
        for multiplier in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            print(f"\n{'='*20} {multiplier}× STD DEV {'='*20}")
            chunker = StandardDeviationSemanticChunker(embeddings, multiplier)
            chunks = chunker.chunk_text(sample_text)
            print(f"Result: {len(chunks)} chunks created")
    
    elif choice == "3":
        print("\n🔬 IQR MULTIPLIER EXPERIMENTATION")
        print("Testing different IQR multipliers...")
        
        for multiplier in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
            print(f"\n{'='*20} {multiplier}× IQR {'='*20}")
            chunker = IQRSemanticChunker(embeddings, multiplier)
            chunks = chunker.chunk_text(sample_text)
            print(f"Result: {len(chunks)} chunks created")
    
    else:
        print("❌ Invalid choice. Please try again.")


def use_custom_text():
    """Allow user to input their own text for chunking."""
    print("📝 USE YOUR OWN TEXT")
    print("=" * 50)
    print("Enter your text (press Enter twice when finished):")
    print("📝 Tip: Use substantial text with multiple topics for best results")
    print()
    
    lines = []
    while True:
        line = input()
        if line == "" and len(lines) > 0 and lines[-1] == "":
            break
        lines.append(line)
    
    custom_text = "\n".join(lines).strip()
    
    if len(custom_text) < 100:
        print("⚠️  Warning: Text is quite short. Results may not be representative.")
        if input("Continue anyway? (y/n): ").lower() != 'y':
            return
    
    print(f"\n📊 Text length: {len(custom_text)} characters")
    print("Choose chunking method:")
    print("1. Percentile-Based")
    print("2. Standard Deviation-Based") 
    print("3. IQR-Based")
    print("4. Compare all methods")
    
    choice = input("Enter choice (1-4): ").strip()
    
    embeddings = initialize_embeddings()
    
    if choice == "1":
        chunker = PercentileSemanticChunker(embeddings)
        chunks = chunker.chunk_text(custom_text)
    elif choice == "2":
        chunker = StandardDeviationSemanticChunker(embeddings)
        chunks = chunker.chunk_text(custom_text)
    elif choice == "3":
        chunker = IQRSemanticChunker(embeddings)
        chunks = chunker.chunk_text(custom_text)
    elif choice == "4":
        from comparison_analysis import SemanticChunkingComparator
        comparator = SemanticChunkingComparator(embeddings)
        chunks_dict = comparator.compare_all_methods(custom_text)
        comparator.visualize_comparisons(chunks_dict)
    else:
        print("❌ Invalid choice.")


def show_help():
    """Show help and documentation."""
    print("❓ HELP & DOCUMENTATION")
    print("=" * 60)
    print()
    print("🎯 WHAT IS SEMANTIC CHUNKING?")
    print("Semantic chunking divides text based on meaning rather than")
    print("fixed sizes, creating more coherent chunks for AI applications.")
    print()
    print("📋 WHEN TO USE EACH METHOD:")
    print()
    print("🔹 Percentile-Based:")
    print("   • General-purpose chunking")
    print("   • Consistent chunk quality needed")
    print("   • Starting point for most applications")
    print()
    print("🔸 Standard Deviation:")
    print("   • Content with varying semantic density")
    print("   • Adaptive chunking needed")
    print("   • Statistical approach preferred")
    print()
    print("🔺 IQR-Based:")
    print("   • Noisy data with outliers")
    print("   • Robust chunking needed")
    print("   • Consistency over sensitivity")
    print()
    print("🔧 PARAMETER GUIDELINES:")
    print("   • Start with default parameters")
    print("   • Experiment with your specific content")
    print("   • Consider downstream application needs")
    print("   • Use comparative analysis for optimization")
    print()
    print("📚 FILES IN THIS SUITE:")
    print("   • utils.py - Shared utilities and functions")
    print("   • percentile_chunking.py - Percentile-based method")
    print("   • std_dev_chunking.py - Standard deviation method") 
    print("   • iqr_chunking.py - IQR-based method")
    print("   • comparison_analysis.py - Comparative analysis")
    print("   • main.py - This interactive menu system")
    print()


def main():
    """Main function with interactive menu system."""
    print_welcome_banner()
    
    while True:
        try:
            print_main_menu()
            choice = input("Enter your choice (0-8): ").strip()
            
            if choice == "0":
                print("\n👋 Thank you for using the Semantic Chunking Suite!")
                print("🎯 Remember: Choose your chunking method based on your specific use case.")
                print("✨ Happy chunking!")
                break
            
            elif choice == "1":
                print("\n" + "🔹" * 30)
                demonstrate_percentile_chunking()
                input("\nPress Enter to continue...")
            
            elif choice == "2":
                print("\n" + "🔸" * 30)
                demonstrate_std_dev_chunking()
                input("\nPress Enter to continue...")
            
            elif choice == "3":
                print("\n" + "🔺" * 30)
                demonstrate_iqr_chunking()
                input("\nPress Enter to continue...")
            
            elif choice == "4":
                print("\n" + "🔍" * 30)
                run_comprehensive_analysis()
                input("\nPress Enter to continue...")
            
            elif choice == "5":
                print("\n" + "🎯" * 30)
                show_method_info()
                input("\nPress Enter to continue...")
            
            elif choice == "6":
                print("\n" + "🧪" * 30)
                parameter_experimentation()
                input("\nPress Enter to continue...")
            
            elif choice == "7":
                print("\n" + "📝" * 30)
                use_custom_text()
                input("\nPress Enter to continue...")
            
            elif choice == "8":
                print("\n" + "❓" * 30)
                show_help()
                input("\nPress Enter to continue...")
            
            else:
                print("❌ Invalid choice. Please enter a number between 0-8.")
            
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or contact support.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()