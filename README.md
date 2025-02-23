# Aspect-Based Sentiment Analysis (ABSA) 🔍🧠

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)

An advanced NLP system for granular sentiment analysis at aspect level, implementing Aspect Sentiment Pair Extraction (ASPE) through two-stage Deep Learning architecture.

## Key Features ✨
- **Dual-Stage Architecture**: Combines Aspect Term Extraction (ATE) and Aspect Term Sentiment Classification (ATSC)
- **State-of-the-Art Models**: Utilizes DistilBERT for efficient text processing
- **High Accuracy**: 81.73% F1-score on ATE and 79.18% accuracy on ATSC
- **Pre-trained Models**: Ready-to-use models for restaurant domain analysis
- **Streamlit Demo**: Interactive web interface for real-time predictions

## Tech Stack 🛠️
| Component          | Technologies                                                                 |
|--------------------|------------------------------------------------------------------------------|
| **Core NLP**       | DistilBERT, Hugging Face Transformers                                        |
| **Data Processing**| Datasets library, NumPy, Pandas                                              |
| **Evaluation**     | Seqeval, Accuracy Metrics                                                    |
| **Deployment**     | Streamlit, Docker                                                            |

## Usage Example 💻
```python
from transformers import pipeline

# Aspect Term Extraction
ate_classifier = pipeline(
    model="thainq107/abte-restaurants-distilbert-base-uncased",
    aggregation_strategy="simple"
)

# Aspect Sentiment Classification
asc_classifier = pipeline(
    model="thainq107/absa-restaurants-distilbert-base-uncased"
)

# Complete ASPE Pipeline
test_sentence = "The bread is top notch though the service was slow"
aspects = ate_classifier(test_sentence)
results = asc_classifier(f"{test_sentence} [SEP] {' '.join([a['word'] for a in aspects])}")

print(f"Aspect-Sentiment Pairs: {[(aspect['word'], result['label']) for aspect, result in zip(aspects, results)]}")
```

## Dataset 📊
**SemEval-2014 Task 4**: Restaurant Reviews
- Preprocessed dataset available at `thainq107/abte-restaurants` on Hugging Face Hub
- Contains 3,600+ training samples with aspect-level annotations

## Model Training 🏋️♂️
### Aspect Term Extraction
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
```

## Results 📈
| Task       | Metric    | Performance |
|------------|-----------|-------------|
| ATE        | F1-Score  | 81.73%      |
| ATSC       | Accuracy  | 79.18%      |

## License 📄
Distributed under MIT License - see [LICENSE](LICENSE) for details

## References & Acknowledgements 📚
- SemEval-2014 Task 4 Dataset
- Hugging Face Transformers Library
- DistilBERT architecture
- AI Vietnam Research Team
