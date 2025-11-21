"""BERT embeddings utility for O-4 experiments.

This module provides a unified interface for loading BERT embeddings,
adapted from sup_exp_14_bert.py with generalized data structure.
"""

import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import warnings

warnings.filterwarnings('ignore')


def set_torch_seed(seed: int):
    """Set all random seeds for PyTorch determinism."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def load_bert_embeddings(
    model_name: str = 'bert-base-uncased',
    seed: int = 42,
    n_items: int = 50,
    radius: float = 1.0
) -> dict:
    """
    Load BERT embeddings with corresponding spatial coordinates.
    
    Parameters
    ----------
    model_name : str, default='bert-base-uncased'
        Pretrained BERT model name
    seed : int, default=42
        Random seed for reproducibility
    n_items : int, default=50
        Number of words to embed
    radius : float, default=1.0
        Radius for initial circle layout
    
    Returns
    -------
    data : dict
        Dictionary with keys:
        - 'embeddings': ndarray, shape (n_items, 768) - BERT embeddings
        - 'coords': ndarray, shape (n_items, 2) - Initial circle coordinates
        - 'words': list of str - Word list
    
    Notes
    -----
    Adapted from sup_exp_14_bert.py. Uses [CLS] token embeddings
    for single-word representation.
    
    Examples
    --------
    >>> data = load_bert_embeddings(n_items=10, seed=42)
    >>> data['embeddings'].shape
    (10, 768)
    >>> data['coords'].shape
    (10, 2)
    """
    set_torch_seed(seed)
    
    # Default word list (50 concrete nouns from sup_exp_14)
    words = [
        # Animals
        'dog', 'cat', 'bird', 'fish', 'horse', 'elephant', 'lion', 'tiger',
        # Vehicles
        'car', 'bus', 'train', 'plane', 'boat', 'bicycle', 'truck', 'motorcycle',
        # Nature
        'tree', 'flower', 'grass', 'mountain', 'river', 'ocean', 'sun', 'moon',
        # Buildings
        'house', 'building', 'castle', 'bridge', 'tower', 'church', 'school', 'hospital',
        # Objects
        'book', 'chair', 'table', 'lamp', 'clock', 'phone', 'computer', 'camera',
        # Food
        'apple', 'bread', 'rice', 'meat', 'milk', 'coffee', 'tea', 'water',
        # Abstract
        'love', 'time', 'life', 'death'
    ][:n_items]
    
    # Force CPU for determinism
    device = torch.device('cpu')
    
    print(f"  Loading BERT model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    embeddings = []
    print(f"  Extracting embeddings for {len(words)} words...")
    
    with torch.no_grad():
        for i, word in enumerate(words):
            inputs = tokenizer(word, return_tensors='pt').to(device)
            outputs = model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
            embeddings.append(embedding)
            
            if (i + 1) % 10 == 0:
                print(f"    {i + 1}/{len(words)} words processed")
    
    embeddings = np.array(embeddings)
    print(f"  ✅ Embeddings shape: {embeddings.shape}")
    
    # Generate initial circle layout
    rng = np.random.default_rng(seed)
    ordering = rng.permutation(n_items)
    angles = 2 * np.pi * np.arange(n_items) / n_items
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    coords = np.column_stack([x, y])
    
    # Apply random ordering
    ordered_coords = np.zeros_like(coords)
    ordered_coords[ordering] = coords
    
    return {
        'embeddings': embeddings,
        'coords': ordered_coords,
        'words': words
    }
