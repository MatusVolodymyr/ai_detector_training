"""
Standalone multilingual stylometric feature extraction for dataset processing.
Optimized for large-scale batch processing without app dependencies.
"""

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import numpy as np
import torch
import textstat
import spacy
import nltk
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple
from lexicalrichness import LexicalRichness
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
)
import re
from tqdm import tqdm


# Language mappings
LANG2ID: Dict[str, int] = {
    "en": 0,
    "de": 1,
    "fr": 2,
    "es": 3,
    "pt": 4,
}

# Multilingual spaCy models
SPACY_MODELS = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm", 
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "pt": "pt_core_news_sm",
}

# High-importance POS tags based on feature analysis
EXPECTED_POS_TAGS = [
    "PUNCT",  # Highest importance
    "X",      # High importance  
    "AUX",    # Medium importance
    "PRON",   # Medium importance
    "PROPN",  # Medium importance
]

# Global model instances
nlp_models = {}
spacy_sentencizer = None
perplexity_model = None
perplexity_tokenizer = None

# Configuration
class Config:
    enable_perplexity = True
    enable_stylometric = True
    perplexity_batch_size = 8
    perplexity_max_length = 512
    max_sentences = 100
    cpu_mode = False


def initialize_models():
    """Initialize all models for multilingual inference."""
    global nlp_models, spacy_sentencizer, perplexity_model, perplexity_tokenizer
    
    print("🔧 Initializing stylometric models...")
    device = torch.device("cuda" if torch.cuda.is_available() and not Config.cpu_mode else "cpu")
    
    # Download required NLTK data
    try:
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass  # Silent failure for NLTK
    
    # Load language-specific spaCy models
    loaded_langs = []
    for lang, model_name in SPACY_MODELS.items():
        try:
            nlp_models[lang] = spacy.load(
                model_name, disable=["parser", "ner", "lemmatizer", "attribute_ruler"]
            )
            loaded_langs.append(lang)
        except OSError:
            nlp_models[lang] = None
    
    print(f"   ✓ spaCy models loaded: {', '.join(loaded_langs)}")
    
    # Load universal sentence splitter
    try:
        spacy_sentencizer = spacy.load(
            "xx_sent_ud_sm", disable=["tagger", "parser", "ner"]
        )
        print("   ✓ Universal sentence splitter loaded")
    except OSError:
        spacy_sentencizer = None
    
    # Load perplexity model if enabled
    if Config.enable_perplexity:
        try:
            import transformers
            transformers.utils.logging.set_verbosity_error()
            
            perplexity_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            perplexity_model = (
                AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=torch.float16)
                .to(device).eval()
            )
            perplexity_tokenizer.pad_token = perplexity_tokenizer.eos_token
            print(f"   ✓ Perplexity model loaded on {device}")
        except Exception:
            try:
                perplexity_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                perplexity_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
                perplexity_tokenizer.pad_token = perplexity_tokenizer.eos_token
                print(f"   ✓ GPT-2 perplexity model loaded on {device}")
            except Exception:
                perplexity_model = None
                perplexity_tokenizer = None
                Config.enable_perplexity = False
                print("   ⚠️ Perplexity model failed to load")
    
    print("✅ Stylometric models ready!")


def detect_language(text: str) -> str:
    """Detect text language with fallback to English."""
    try:
        from langdetect import detect
        detected_lang = detect(text)
        
        language_mapping = {
            "en": "en", "de": "de", "fr": "fr", "es": "es", "pt": "pt",
            "ca": "es", "it": "en", "nl": "en",  # Fallbacks
        }
        
        return language_mapping.get(detected_lang, "en")
    except Exception:
        return "en"


def clean_text(text: str) -> str:
    """Clean markdown, HTML tags, and extra whitespace."""
    # Remove HTML tags
    text = re.sub(r"</?(?:p|div|span|br|hr|img|a|h[1-6]|ul|ol|li|table|tr|td|th|thead|tbody|strong|em|b|i|u|script|style|head|body|html|meta|link|title|form|input|button|label|select|option|textarea)(?:\s[^>]*)?>|<!--.*?-->", "", text, flags=re.IGNORECASE)
    # Remove markdown patterns
    text = re.sub(r"\!\[.*?\]\(.*?\)", "", text)  # Images
    text = re.sub(r"\[([^\]]+)\]\((.*?)\)", r"\1", text)  # Links
    text = re.sub(r"`{1,3}[^`]+`{1,3}", "", text)  # Code
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # Bold
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # Italics
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> List[str]:
    """Multilingual sentence splitting."""
    try:
        if spacy_sentencizer is not None:
            doc = spacy_sentencizer(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            return nltk.sent_tokenize(text)
    except Exception:
        return [text]


def create_windows(text: str, max_sentences: int = Config.max_sentences) -> Dict[str, List[str]]:
    """Create different window sizes for multiscale analysis."""
    if not text or not text.strip():
        return {
            "window_1": [""],
            "window_3": [""],
            "window_5": [""],
            "language": ["en"]
        }
    
    # Clean and detect language
    cleaned_text = clean_text(text)
    language = detect_language(cleaned_text)
    
    # Split into sentences
    sentences = split_sentences(cleaned_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return {
            "window_1": [""],
            "window_3": [""],
            "window_5": [""],
            "language": [language]
        }
    
    # Limit number of sentences for performance
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
    
    # Create windows
    windows = {
        "window_1": sentences,
        "window_3": [],
        "window_5": [],
        "language": [language] * len(sentences)
    }
    
    # Create 3-sentence windows
    for i in range(len(sentences)):
        start_idx = max(0, i - 1)
        end_idx = min(len(sentences), i + 2)
        window_3_text = " ".join(sentences[start_idx:end_idx])
        windows["window_3"].append(window_3_text)
    
    # Create 5-sentence windows
    for i in range(len(sentences)):
        start_idx = max(0, i - 2)
        end_idx = min(len(sentences), i + 3)
        window_5_text = " ".join(sentences[start_idx:end_idx])
        windows["window_5"].append(window_5_text)
    
    return windows


def create_multiscale_windows(text: str) -> Dict[str, List[str]]:
    """Create multiscale windows for text analysis (compatibility function)."""
    return create_windows(text)


def extract_readability_features(text: str) -> Dict[str, float]:
    """Extract readability features optimized for speed."""
    if not text or not text.strip():
        return {"smog_index": 0.0, "ari": 0.0}
    
    try:
        # Only extract high-importance features based on analysis
        return {
            "smog_index": float(textstat.smog_index(text)),
            "ari": float(textstat.automated_readability_index(text)),
        }
    except Exception:
        return {"smog_index": 0.0, "ari": 0.0}


def extract_lexical_features(text: str) -> Dict[str, float]:
    """Extract lexical diversity features."""
    if not text or not text.strip():
        return {"type_token_ratio": 0.0}
    
    try:
        words = text.lower().split()
        if len(words) == 0:
            return {"type_token_ratio": 0.0}
        
        # Simple type-token ratio calculation
        unique_words = len(set(words))
        total_words = len(words)
        ttr = unique_words / total_words if total_words > 0 else 0.0
        
        return {"type_token_ratio": float(ttr)}
    except Exception:
        return {"type_token_ratio": 0.0}


def extract_pos_features(text: str, language: str = "en") -> Dict[str, float]:
    """Extract POS distribution features for a single text."""
    if not text or not text.strip():
        return {f"pos_{tag}": 0.0 for tag in EXPECTED_POS_TAGS}
    
    try:
        nlp_model = nlp_models.get(language, nlp_models.get("en"))
        if nlp_model is None:
            return {f"pos_{tag}": 0.0 for tag in EXPECTED_POS_TAGS}
        
        doc = nlp_model(text)
        pos_counts = Counter([token.pos_ for token in doc])
        total = sum(pos_counts.values())
        
        return {
            f"pos_{tag}": pos_counts.get(tag, 0) / total if total > 0 else 0.0
            for tag in EXPECTED_POS_TAGS
        }
    except Exception:
        return {f"pos_{tag}": 0.0 for tag in EXPECTED_POS_TAGS}


def extract_pos_features_batch(texts: List[str], languages: List[str]) -> List[Dict[str, float]]:
    """Extract POS distribution features for a batch of texts efficiently."""
    if not texts:
        return []
    
    # Group texts by language for efficient batch processing
    lang_groups = {}
    for i, (text, lang) in enumerate(zip(texts, languages)):
        if lang not in lang_groups:
            lang_groups[lang] = []
        lang_groups[lang].append((i, text))
    
    # Initialize results array
    results: List[Dict[str, float]] = [
        {f"pos_{tag}": 0.0 for tag in EXPECTED_POS_TAGS} for _ in range(len(texts))
    ]
    
    # Process each language group in batch
    for lang, text_indices in lang_groups.items():
        nlp_model = nlp_models.get(lang, nlp_models.get("en"))
        if nlp_model is None:
            continue
        
        try:
            batch_texts = [text for _, text in text_indices]
            docs = list(nlp_model.pipe(batch_texts, disable=["parser", "ner", "lemmatizer"]))
            
            for (original_idx, _), doc in zip(text_indices, docs):
                pos_counts = Counter([token.pos_ for token in doc])
                total = sum(pos_counts.values())
                
                results[original_idx] = {
                    f"pos_{tag}": pos_counts.get(tag, 0) / total if total > 0 else 0.0
                    for tag in EXPECTED_POS_TAGS
                }
                
        except Exception as e:
            print(f"Warning: Batch POS processing failed for {lang}: {e}")
            for idx, text in text_indices:
                results[idx] = extract_pos_features(text, lang)
    
    return results


def extract_punctuation_features(text: str) -> Dict[str, float]:
    """Extract punctuation density features."""
    if not text or not text.strip():
        return {"punct_density": 0.0, "exclamations": 0, "ellipsis": 0}
    
    try:
        sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))
        punct_count = len([c for c in text if c in '.!?'])
        
        return {
            "punct_density": punct_count / sentence_count,
            "exclamations": text.count('!'),
            "ellipsis": text.count('...'),
        }
    except Exception:
        return {"punct_density": 0.0, "exclamations": 0, "ellipsis": 0}


def extract_sentence_length_features(text: str) -> Dict[str, float]:
    """Extract sentence length features using fast word counting."""
    try:
        sentences = split_sentences(text)
        if not sentences:
            return {"avg_sentence_length": 0.0, "sentence_length_std": 0.0}
        
        lengths = [len(s.split()) for s in sentences]
        if not lengths:
            return {"avg_sentence_length": 0.0, "sentence_length_std": 0.0}
        
        if len(lengths) == 1:
            return {
                "avg_sentence_length": float(lengths[0]),
                "sentence_length_std": 0.0,
            }
        
        return {
            "avg_sentence_length": float(np.mean(lengths)),
            "sentence_length_std": float(np.std(lengths, ddof=1)),
        }
    except Exception:
        return {"avg_sentence_length": 0.0, "sentence_length_std": 0.0}


@torch.inference_mode()
def calculate_perplexity_batch(texts: List[str], batch_size: int = Config.perplexity_batch_size) -> List[float]:
    """Calculate perplexity for a batch of texts efficiently."""
    if perplexity_model is None or perplexity_tokenizer is None:
        return [0.0] * len(texts)
    
    perplexities = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        
        try:
            device = perplexity_model.device
            encoded = perplexity_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=Config.perplexity_max_length,
            )
            
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            outputs = perplexity_model(input_ids, attention_mask=attention_mask)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)
            
            target_log_probs = (
                log_probs[:, :-1]
                .gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
                .squeeze(-1)
            )
            
            mask = attention_mask[:, 1:]
            seq_log_probs = (target_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            batch_perplexities = torch.exp(-seq_log_probs).cpu().tolist()
            
            # Handle NaN/inf values
            batch_perplexities = [
                0.0 if np.isnan(p) or np.isinf(p) else float(p)
                for p in batch_perplexities
            ]
            
            perplexities.extend(batch_perplexities)
            
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            perplexities.extend([0.0] * len(batch_texts))
    
    return perplexities


def extract_features_batch(texts: List[str], languages: List[str]) -> Dict[str, List[Dict[str, float]]]:
    """Extract all non-perplexity features in batch for better performance."""
    if not texts:
        return {
            "readability": [],
            "lexical": [],
            "punctuation": [],
            "sentence_length": [],
            "pos": []
        }
    
    batch_features = {
        "readability": [],
        "lexical": [],
        "punctuation": [],
        "sentence_length": [],
        "pos": []
    }
    
    # Process simple features
    for text in texts:
        batch_features["readability"].append(extract_readability_features(text))
        batch_features["lexical"].append(extract_lexical_features(text))
        batch_features["punctuation"].append(extract_punctuation_features(text))
        batch_features["sentence_length"].append(extract_sentence_length_features(text))
    
    # POS features in batch (most expensive operation)
    pos_features_batch = extract_pos_features_batch(texts, languages)
    batch_features["pos"] = pos_features_batch
    
    return batch_features


def extract_all_features(windows_dict: Dict[str, Any], include_perplexity: bool = True) -> Dict[str, List[float]]:
    """Extract all stylometric features using optimized batch processing."""
    features = {}
    languages = windows_dict.get("language", ["en"] * len(windows_dict["window_1"]))
    
    for window_type in ["window_1", "window_3", "window_5"]:
        if window_type not in windows_dict:
            continue
        
        texts = windows_dict[window_type]
        
        # Initialize feature lists
        features[f"{window_type}_smog_index"] = []
        features[f"{window_type}_ari"] = []
        features[f"{window_type}_type_token_ratio"] = []
        features[f"{window_type}_punct_density"] = []
        features[f"{window_type}_exclamations"] = []
        features[f"{window_type}_ellipsis"] = []
        features[f"{window_type}_avg_sentence_length"] = []
        features[f"{window_type}_sentence_length_std"] = []
        
        for tag in EXPECTED_POS_TAGS:
            features[f"{window_type}_pos_{tag}"] = []
        
        # Calculate perplexity if requested
        if include_perplexity:
            perplexities = calculate_perplexity_batch(texts)
            features[f"{window_type}_perplexity"] = perplexities
        
        # Use batch processing for better performance
        batch_features = extract_features_batch(texts, languages)
        
        # Distribute batch results to individual feature arrays
        for i in range(len(texts)):
            # Readability
            readability = batch_features["readability"][i]
            features[f"{window_type}_smog_index"].append(readability["smog_index"])
            features[f"{window_type}_ari"].append(readability["ari"])
            
            # Lexical diversity
            lexical = batch_features["lexical"][i]
            features[f"{window_type}_type_token_ratio"].append(lexical["type_token_ratio"])
            
            # Punctuation
            punct = batch_features["punctuation"][i]
            features[f"{window_type}_punct_density"].append(punct["punct_density"])
            features[f"{window_type}_exclamations"].append(punct["exclamations"])
            features[f"{window_type}_ellipsis"].append(punct["ellipsis"])
            
            # Sentence length
            sent_len = batch_features["sentence_length"][i]
            features[f"{window_type}_avg_sentence_length"].append(sent_len["avg_sentence_length"])
            features[f"{window_type}_sentence_length_std"].append(sent_len["sentence_length_std"])
            
            # POS features from batch
            pos_features = batch_features["pos"][i]
            for tag in EXPECTED_POS_TAGS:
                features[f"{window_type}_pos_{tag}"].append(pos_features[f"pos_{tag}"])
    
    return features


def process_example(example):
    """Process a single example from the dataset to extract stylometric features."""
    # Extract windows dict from example
    windows_dict = {
        "window_1": [example["window_1"]],
        "window_3": [example["window_3"]],
        "window_5": [example["window_5"]],
        "language": [example["language"]]
    }
    
    # Extract all stylometric features
    features = extract_all_features(windows_dict, include_perplexity=Config.enable_perplexity)
    
    # Convert lists to single values (since we have only one example)
    single_features = {k: v[0] if v else 0.0 for k, v in features.items()}
    
    # Add language_id
    single_features["language_id"] = LANG2ID.get(example["language"], -1)
    
    return {**example, **single_features}


def process_dataset_batch(examples):
    """Process a batch of examples from the dataset."""
    # Prepare batch data
    batch_windows = {
        "window_1": examples["window_1"],
        "window_3": examples["window_3"], 
        "window_5": examples["window_5"],
        "language": examples["language"]
    }
    
    # Extract features for the entire batch
    batch_features = extract_all_features(batch_windows, include_perplexity=Config.enable_perplexity)
    
    # Add language_id
    batch_features["language_id"] = [LANG2ID.get(lang, -1) for lang in examples["language"]]
    
    return batch_features


# Example usage and testing
if __name__ == "__main__":
    # Test the functions
    initialize_models()
    
    test_text = """This is a sample text for testing. It has multiple sentences. 
    The system should extract various features from this text efficiently."""
    
    print("Testing individual text processing...")
    windows = create_multiscale_windows(test_text)
    features = extract_all_features(windows)
    
    print(f"Created {len(windows['window_1'])} windows")
    print(f"Extracted {len(features)} feature types")
    print("Sample features:", list(features.keys())[:5])
