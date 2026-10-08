import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import CLEANED_DATA_PATH, IMAGES_DIR, CLASS_MAP
from src.feature_extraction import FeatureExtractor

# Set consistent matplotlib styling
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#4a5568'
plt.rcParams['axes.linewidth'] = 1.2

class DataVisualizer:
    """
    Generates dataset visualization plots and textual summary reports.
    """
    def __init__(self, data_path: str = CLEANED_DATA_PATH, output_dir: str = IMAGES_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        if os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
        else:
            from src.preprocessing import TextPreprocessor
            preprocessor = TextPreprocessor()
            self.df = preprocessor.preprocess_dataset()

    def plot_class_distribution(self):
        """
        Plots class distribution bar & pie charts.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0f172a')
        
        class_counts = self.df['class'].value_counts().sort_index()
        labels = [CLASS_MAP[c] for c in class_counts.index]
        colors = ['#ef4444', '#f59e0b', '#10b981']  # Crimson, Amber, Emerald
        
        # Bar chart
        bars = axes[0].bar(labels, class_counts.values, color=colors, edgecolor='#334155', linewidth=1.5)
        axes[0].set_title('Target Class Counts', fontsize=14, fontweight='bold', pad=15, color='#f8fafc')
        axes[0].set_ylabel('Number of Instances', fontsize=12, color='#cbd5e1')
        axes[0].tick_params(colors='#cbd5e1')
        axes[0].grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(f'{height:,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', color='#f8fafc', fontweight='bold')
            
        # Pie chart
        wedges, texts, autotexts = axes[1].pie(
            class_counts.values, labels=labels, autopct='%1.1f%%',
            startangle=140, colors=colors, explode=(0.05, 0.02, 0.02),
            textprops={'color': '#f8fafc', 'fontsize': 11}
        )
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        axes[1].set_title('Target Class Percentage Distribution', fontsize=14, fontweight='bold', pad=15, color='#f8fafc')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'class_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print(f"[SUCCESS] Saved class distribution plot to {save_path}")

    def plot_text_length_distribution(self):
        """
        Plots character length distribution before vs after cleaning.
        """
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0f172a')
        
        sns.kdeplot(self.df['raw_length'], ax=ax, label='Raw Tweet Length', color='#ef4444', fill=True, alpha=0.3, linewidth=2)
        sns.kdeplot(self.df['cleaned_length'], ax=ax, label='Cleaned Tweet Length', color='#10b981', fill=True, alpha=0.3, linewidth=2)
        
        ax.set_title('Text Character Length Distribution (Before vs After Preprocessing)', fontsize=13, fontweight='bold', color='#f8fafc', pad=15)
        ax.set_xlabel('Character Count', fontsize=11, color='#cbd5e1')
        ax.set_ylabel('Density', fontsize=11, color='#cbd5e1')
        ax.tick_params(colors='#cbd5e1')
        ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc')
        ax.grid(True, linestyle='--', alpha=0.2)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'text_length_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print(f"[SUCCESS] Saved text length distribution plot to {save_path}")

    def plot_top_ngrams_per_class(self):
        """
        Plots top 10 unigrams for each class.
        """
        extractor = FeatureExtractor()
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0f172a')
        colors = ['#ef4444', '#f59e0b', '#10b981']
        
        for idx in range(3):
            sub_df = self.df[self.df['class'] == idx]
            top_words = extractor.get_top_ngrams(sub_df['cleaned_tweet'], n=10)
            words = [w[0] for w in top_words[::-1]]
            counts = [w[1] for w in top_words[::-1]]
            
            axes[idx].barh(words, counts, color=colors[idx], edgecolor='#334155')
            axes[idx].set_title(f'Top Words: {CLASS_MAP[idx]}', fontsize=12, fontweight='bold', color='#f8fafc', pad=10)
            axes[idx].set_xlabel('Frequency', fontsize=10, color='#cbd5e1')
            axes[idx].tick_params(colors='#cbd5e1')
            axes[idx].grid(axis='x', linestyle='--', alpha=0.3)
            
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'top_ngrams.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print(f"[SUCCESS] Saved top n-grams plot to {save_path}")

    def plot_confusion_matrix(self, cm: np.ndarray, labels: list = None):
        """
        Plots confusion matrix heatmap.
        """
        if labels is None:
            labels = [CLASS_MAP[i] for i in range(3)]
            
        fig, ax = plt.subplots(figsize=(7, 6), facecolor='#0f172a')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,
                    cbar=False, ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
        
        ax.set_title('Model Confusion Matrix', fontsize=14, fontweight='bold', color='#f8fafc', pad=15)
        ax.set_xlabel('Predicted Label', fontsize=12, color='#cbd5e1')
        ax.set_ylabel('True Label', fontsize=12, color='#cbd5e1')
        ax.tick_params(colors='#cbd5e1')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print(f"[SUCCESS] Saved confusion matrix plot to {save_path}")

    def generate_visualization_report(self) -> str:
        """
        Returns textual conclusion report of data visualizations.
        """
        counts = self.df['class'].value_counts()
        total = len(self.df)
        p0 = (counts.get(0, 0) / total) * 100
        p1 = (counts.get(1, 0) / total) * 100
        p2 = (counts.get(2, 0) / total) * 100

        report = f"""================================================================================
                 DATA VISUALIZATION & EXPLORATORY ANALYSIS REPORT
================================================================================
1. TARGET CLASS DISTRIBUTION & IMBALANCE:
   - Total Tweets Analyzed: {total:,}
   - Class 1 (Offensive Language) : {counts.get(1, 0):>6,} ({p1:.1f}%) [Majority Class]
   - Class 2 (Neither / Neutral)   : {counts.get(2, 0):>6,} ({p2:.1f}%)
   - Class 0 (Hate Speech)         : {counts.get(0, 0):>6,} ({p0:.1f}%) [Minority Class]

2. EXPLORATORY INSIGHTS & CONCLUSION:
   - Severe Imbalance Detected: Offensive language constitutes {p1:.1f}% of the dataset, whereas explicit hate speech is only {p0:.1f}%.
   - Implication for ML Modeling: Standard accuracy alone will be misleading due to class imbalance. The machine learning model requires class weighting ('class_weight="balanced"') and evaluation using Macro F1-Score and Precision-Recall per class.
   - Distinct Keyword Boundaries:
     * Hate Speech exhibits specific racial/ethnic slurs and targeted slurs.
     * Offensive Language contains heavy profanity used casually or defensively without hate speech context.
     * Neither Class contains conversational stop words, neutral terms, and news/hashtag phrasing.
================================================================================
"""
        return report

if __name__ == "__main__":
    visualizer = DataVisualizer()
    visualizer.plot_class_distribution()
    visualizer.plot_text_length_distribution()
    visualizer.plot_top_ngrams_per_class()
    report = visualizer.generate_visualization_report()
    print(report)
