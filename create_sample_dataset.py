"""
Example script to create a sample dataset for testing the pipeline
"""

from datasets import Dataset
import random

# Set seed for reproducibility
random.seed(42)

# Generate sample data
def generate_sample_text(is_ai: bool) -> str:
    """Generate sample text that mimics human or AI writing"""
    if is_ai:
        templates = [
            "In conclusion, {topic} is important because {reason}. Furthermore, {additional}.",
            "The significance of {topic} cannot be overstated. It plays a crucial role in {reason}.",
            "When considering {topic}, one must acknowledge that {reason}. This is particularly evident in {additional}.",
            "{topic} represents a fundamental aspect of modern society. Its impact on {reason} is undeniable.",
        ]
    else:
        templates = [
            "I think {topic} matters cuz {reason}... and yeah {additional}",
            "tbh {topic} is super important! Like, {reason} and stuff like that.",
            "{topic}? well... {reason}. Also {additional} lol",
            "So basically {topic} is a big deal because {reason}. Pretty wild right?",
        ]
    
    topics = ["artificial intelligence", "climate change", "education", "technology", "healthcare"]
    reasons = ["it affects everyone", "we need solutions", "future depends on it", "society needs it"]
    additional = ["lots of research shows this", "experts agree", "data supports it", "evidence indicates"]
    
    template = random.choice(templates)
    return template.format(
        topic=random.choice(topics),
        reason=random.choice(reasons),
        additional=random.choice(additional)
    )

# Generate dataset
print("🔨 Creating sample dataset...")

num_samples = 1000  # Small dataset for testing
texts = []
labels = []

for i in range(num_samples):
    is_ai = i % 2 == 0  # Alternate between AI and human
    texts.append(generate_sample_text(is_ai))
    labels.append(1 if is_ai else 0)

# Create dataset
data = {
    'text': texts,
    'label': labels,
}

dataset = Dataset.from_dict(data)

# Save dataset
output_path = 'sample_training_data.arrow'
dataset.save_to_disk(output_path)

print(f"✅ Sample dataset created!")
print(f"   Path: {output_path}")
print(f"   Size: {len(dataset)} examples")
print(f"   AI texts: {sum(labels)}")
print(f"   Human texts: {len(labels) - sum(labels)}")
print(f"\nYou can now test the pipeline with:")
print(f"   python main.py --config config.yaml")
print(f"\nMake sure to update config.yaml:")
print(f"   dataset.path: '{output_path}'")
