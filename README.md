# 🧠 LangChain Chunking 

- 📊 Interactive menu system and comparative analysis
- 🔧 Customizable parameters for each method
- 📈 Statistical analysis and visualizations
Implementation of semantic and recursive chunking methods in LangChain for text processing and RAG applications.

## � Project Purpose

This project demonstrates different text chunking strategies essential for building effective Retrieval-Augmented Generation (RAG) systems and document processing pipelines. Proper text chunking is crucial for:

- **Better Information Retrieval** - Semantic chunks preserve context and meaning
- **Improved RAG Performance** - Relevant information stays together in chunks
- **Flexible Text Processing** - Different methods suit different document types
- **Educational Learning** - Compare and understand various chunking approaches

Choose the right chunking method based on your text type and application needs!

## 🎯 Chunking Methods

### Semantic Chunking:
- **Percentile-Based** - Uses percentile thresholds for semantic similarity
- **Standard Deviation** - Statistical deviation for chunk boundaries  
- **IQR-Based** - Quartile analysis for robust chunking

### Traditional Chunking:
- **Recursive Character** - Hierarchical text splitting with multiple separators

## ✨ Features

- � Interactive menu system and comparative analysis
- 📓 Jupyter notebook with step-by-step examples  
- 🔧 Customizable parameters for each method
- � Statistical analysis and visualizations

## 🚀 Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start

```bash
# Interactive menu (recommended)
cd Semantic_Chunking
python main.py

# Individual methods
python percentile_chunking.py
python std_dev_chunking.py  
python iqr_chunking.py

# Recursive chunking
python Recursive_Chunking.py
```

## 💡 Usage Example

```python
from percentile_chunking import PercentileSemanticChunker

chunker = PercentileSemanticChunker()
chunks = chunker.chunk_text("Your text here", percentile=75)
chunker.display_chunks(chunks)
```

## 📁 File Structure

```
Chunking/
├── README.md & requirements.txt  
├── Recursive_Chunking.py         # Traditional recursive chunking
└── Semantic_Chunking/
    ├── main.py                   # Interactive menu
    ├── utils.py                  # Shared utilities  
    ├── percentile_chunking.py    # Percentile method
    ├── std_dev_chunking.py       # Standard deviation method
    ├── iqr_chunking.py          # IQR method
    └── comparison_analysis.py    # Analysis tools
```

## � Key Parameters

- **Percentile**: 50, 75, 90, 95 (higher = fewer chunks)
- **Std Dev Multiplier**: 0.5, 1.0, 1.5, 2.0 (higher = fewer chunks)  
- **IQR**: Automatic quartile calculation

---

**Happy Chunking! 🚀**