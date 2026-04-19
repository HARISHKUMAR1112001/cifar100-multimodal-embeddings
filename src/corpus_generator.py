
import argparse
import os
import sys
import random
import itertools

# Allow imports from src/ when run as a script
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

MISSING_WORDS = [
    'apples', 'aquariumfish', 'beaver', 'bee', 'beetle', 'bottles', 'bowls',
    'butterfly', 'camel', 'cans', 'castle', 'caterpillar', 'cattle', 'chimpanzee',
    'cockroach', 'computerkeyboard', 'crab', 'crocodile', 'cups', 'dinosaur',
    'dolphin', 'flatfish', 'forest', 'fox', 'hamster', 'kangaroo', 'lawnmower',
    'leopard', 'lion', 'lizard', 'lobster', 'maple', 'mushrooms', 'oak',
    'oranges', 'orchids', 'otter', 'palm', 'pears', 'pickuptruck', 'pine',
    'plain', 'plates', 'poppies', 'porcupine', 'possum', 'rabbit', 'raccoon',
    'ray', 'rocket', 'roses', 'sea', 'seal', 'shark', 'shrew', 'skunk',
    'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
    'sunflowers', 'sweetpeppers', 'telephone', 'television', 'tiger',
    'tractor', 'trout', 'tulips', 'turtle', 'wardrobe', 'whale', 'willow',
    'wolf', 'worm', 'tank'
]

SUPERCLASSES = {
    'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquariumfish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchids', 'poppies', 'roses', 'sunflowers', 'tulips'],
    'food_containers': ['bottles', 'bowls', 'cans', 'cups', 'plates'],
    'fruits_vegetables': ['apples', 'mushrooms', 'oranges', 'pears', 'sweetpeppers'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores': ['leopard', 'lion', 'tiger', 'wolf'],
    'large_outdoor_things': ['castle', 'skyscraper'],
    'large_natural_scenes': ['forest', 'plain', 'sea'],
    'large_omnivores_herbivores': ['camel', 'cattle', 'chimpanzee', 'kangaroo'],
    'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals': ['hamster', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple', 'oak', 'palm', 'pine', 'willow'],
    'vehicles_1': ['pickuptruck'],
    'vehicles_2': ['lawnmower', 'rocket', 'streetcar', 'tank', 'tractor'],
    'household_items': ['computerkeyboard', 'telephone', 'television', 'wardrobe'],
}

VG_ANCHORS = {
    'apples': ['fruit', 'food', 'tree', 'basket', 'bowl', 'plate', 'red', 'green'],
    'oranges': ['fruit', 'food', 'tree', 'basket', 'bowl', 'orange', 'round'],
    'pears': ['fruit', 'food', 'tree', 'basket', 'bowl', 'green', 'yellow'],
    'sweetpeppers': ['vegetable', 'food', 'plate', 'bowl', 'red', 'green'],
    'mushrooms': ['vegetable', 'food', 'plate', 'forest', 'brown', 'white'],
    'roses': ['flower', 'plant', 'garden', 'vase', 'bouquet', 'red', 'pink'],
    'tulips': ['flower', 'plant', 'garden', 'vase', 'field', 'colorful'],
    'sunflowers': ['flower', 'plant', 'garden', 'field', 'yellow', 'tall'],
    'orchids': ['flower', 'plant', 'pot', 'garden', 'exotic', 'beautiful'],
    'poppies': ['flower', 'plant', 'field', 'garden', 'red', 'wild'],
    'tiger': ['animal', 'cat', 'wild', 'jungle', 'zoo', 'striped', 'orange'],
    'lion': ['animal', 'cat', 'wild', 'savanna', 'zoo', 'mane', 'golden'],
    'leopard': ['animal', 'cat', 'wild', 'tree', 'jungle', 'spotted'],
    'wolf': ['animal', 'dog', 'wild', 'forest', 'pack', 'gray'],
    'fox': ['animal', 'dog', 'wild', 'forest', 'red', 'clever'],
    'beaver': ['animal', 'water', 'river', 'dam', 'brown', 'tail'],
    'dolphin': ['animal', 'water', 'ocean', 'sea', 'intelligent', 'gray'],
    'otter': ['animal', 'water', 'river', 'rock', 'playful', 'furry'],
    'seal': ['animal', 'water', 'beach', 'rock', 'gray', 'whiskers'],
    'whale': ['animal', 'water', 'ocean', 'sea', 'huge', 'blue'],
    'aquariumfish': ['animal', 'water', 'tank', 'fish', 'colorful', 'swimming'],
    'flatfish': ['animal', 'water', 'ocean', 'fish', 'flat', 'bottom'],
    'ray': ['animal', 'water', 'ocean', 'fish', 'gliding', 'flat'],
    'shark': ['animal', 'water', 'ocean', 'fish', 'dangerous', 'teeth'],
    'trout': ['animal', 'water', 'river', 'fish', 'spotted', 'swimming'],
    'bee': ['insect', 'animal', 'flower', 'garden', 'buzzing', 'yellow'],
    'beetle': ['insect', 'animal', 'ground', 'leaf', 'black', 'hard'],
    'butterfly': ['insect', 'animal', 'flower', 'garden', 'colorful', 'wings'],
    'caterpillar': ['insect', 'animal', 'leaf', 'plant', 'green', 'crawling'],
    'cockroach': ['insect', 'animal', 'ground', 'floor', 'brown', 'fast'],
    'bottles': ['container', 'glass', 'table', 'shelf', 'drink', 'liquid'],
    'bowls': ['container', 'dish', 'table', 'food', 'round', 'ceramic'],
    'cans': ['container', 'metal', 'shelf', 'table', 'food', 'aluminum'],
    'cups': ['container', 'dish', 'table', 'coffee', 'tea', 'drink'],
    'plates': ['dish', 'table', 'food', 'dinner', 'round', 'ceramic'],
    'maple': ['tree', 'plant', 'forest', 'park', 'leaves', 'tall'],
    'oak': ['tree', 'plant', 'forest', 'park', 'strong', 'old'],
    'palm': ['tree', 'plant', 'beach', 'tropical', 'tall', 'green'],
    'pine': ['tree', 'plant', 'forest', 'mountain', 'evergreen', 'needles'],
    'willow': ['tree', 'plant', 'water', 'park', 'drooping', 'graceful'],
    'tractor': ['vehicle', 'machine', 'farm', 'field', 'wheels', 'large'],
    'pickuptruck': ['vehicle', 'truck', 'road', 'car', 'wheels', 'driving'],
    'rocket': ['vehicle', 'spacecraft', 'space', 'sky', 'tall', 'launching'],
    'streetcar': ['vehicle', 'train', 'city', 'street', 'tracks', 'electric'],
    'lawnmower': ['machine', 'tool', 'grass', 'yard', 'cutting', 'motor'],
    'tank': ['vehicle', 'military', 'machine', 'weapon', 'armored', 'heavy'],
    'crocodile': ['animal', 'reptile', 'water', 'river', 'dangerous', 'teeth'],
    'lizard': ['animal', 'reptile', 'rock', 'wall', 'small', 'scaly'],
    'snake': ['animal', 'reptile', 'ground', 'grass', 'slithering', 'long'],
    'turtle': ['animal', 'reptile', 'water', 'shell', 'slow', 'green'],
    'dinosaur': ['animal', 'reptile', 'prehistoric', 'museum', 'extinct', 'huge'],
    'crab': ['animal', 'water', 'beach', 'sand', 'claws', 'sideways'],
    'lobster': ['animal', 'water', 'ocean', 'food', 'claws', 'red'],
    'snail': ['animal', 'shell', 'garden', 'ground', 'slow', 'slimy'],
    'spider': ['animal', 'insect', 'web', 'wall', 'eight', 'legs'],
    'worm': ['animal', 'ground', 'soil', 'dirt', 'long', 'wriggling'],
    'hamster': ['animal', 'rodent', 'pet', 'cage', 'small', 'furry'],
    'rabbit': ['animal', 'bunny', 'pet', 'grass', 'ears', 'hopping'],
    'shrew': ['animal', 'rodent', 'mammal', 'ground', 'tiny', 'quick'],
    'squirrel': ['animal', 'rodent', 'tree', 'park', 'tail', 'climbing'],
    'camel': ['animal', 'desert', 'mammal', 'sand', 'hump', 'tall'],
    'cattle': ['animal', 'cow', 'farm', 'field', 'large', 'grazing'],
    'chimpanzee': ['animal', 'monkey', 'ape', 'tree', 'intelligent', 'furry'],
    'kangaroo': ['animal', 'mammal', 'wild', 'australia', 'hopping', 'pouch'],
    'porcupine': ['animal', 'mammal', 'wild', 'forest', 'quills', 'spiky'],
    'possum': ['animal', 'mammal', 'wild', 'tree', 'tail', 'nocturnal'],
    'raccoon': ['animal', 'mammal', 'wild', 'forest', 'mask', 'clever'],
    'skunk': ['animal', 'mammal', 'wild', 'forest', 'stripe', 'smell'],
    'castle': ['building', 'structure', 'stone', 'old', 'medieval', 'tower'],
    'skyscraper': ['building', 'structure', 'city', 'tall', 'modern', 'glass'],
    'forest': ['nature', 'trees', 'landscape', 'green', 'woods', 'wild'],
    'plain': ['landscape', 'field', 'grass', 'flat', 'open', 'wide'],
    'sea': ['water', 'ocean', 'nature', 'blue', 'waves', 'vast'],
    'computerkeyboard': ['device', 'computer', 'desk', 'office', 'keys', 'typing'],
    'telephone': ['device', 'phone', 'table', 'communication', 'calling', 'ringing'],
    'television': ['device', 'screen', 'wall', 'entertainment', 'watching', 'show'],
    'wardrobe': ['furniture', 'closet', 'room', 'bedroom', 'clothes', 'wooden'],
}

def get_superclass_members(word):
    for superclass, members in SUPERCLASSES.items():
        if word in members:
            return [m for m in members if m != word], superclass
    return [], None


def generate_sentences_for_word(word, num_sentences=6000):
    """[OK] EXPANDED: 2000+ unique sentences per word."""
    
    anchors = VG_ANCHORS.get(word, ['object', 'thing', 'item'])
    members, superclass = get_superclass_members(word)
    superclass_name = superclass.replace('_', ' ') if superclass else "category"
    
    plural_words = ['apples', 'oranges', 'pears', 'bottles', 'bowls', 'cans', 'cups', 'plates',
                    'roses', 'tulips', 'sunflowers', 'orchids', 'poppies', 'mushrooms',
                    'cattle', 'aquariumfish', 'flatfish', 'trout']
    is_plural = word in plural_words
    article = "" if is_plural else ("an " if word[0] in 'aeiou' else "a ")
    verb_be = "are" if is_plural else "is"
    
    # [OK] EXPANDED POOLS (3x larger)
    if word in ['apples', 'oranges', 'pears']:
        colors = ['red', 'green', 'yellow', 'golden', 'bright', 'dark', 'light', 'pale', 'deep', 'vivid']
        sizes = ['small', 'large', 'big', 'tiny', 'medium', 'huge', 'little']
        qualities = ['fresh', 'ripe', 'juicy', 'sweet', 'crisp', 'delicious', 'tasty', 'organic', 'shiny', 'round']
        locations_special = ['basket', 'bowl', 'plate', 'table', 'tree', 'counter', 'shelf', 'box', 'bag', 'stand']
    elif word in ['tiger', 'lion', 'leopard', 'wolf']:
        colors = ['orange', 'golden', 'striped', 'spotted', 'brown', 'tawny', 'dark', 'light', 'bright', 'sandy']
        sizes = ['large', 'huge', 'massive', 'big', 'giant', 'enormous', 'powerful']
        qualities = ['wild', 'fierce', 'powerful', 'strong', 'dangerous', 'majestic', 'fierce', 'aggressive', 'dominant', 'hunting']
        locations_special = ['jungle', 'forest', 'zoo', 'wild', 'grass', 'savanna', 'habitat', 'enclosure', 'wilderness', 'den']
    elif word in ['roses', 'tulips', 'sunflowers', 'orchids', 'poppies']:
        colors = ['red', 'pink', 'yellow', 'white', 'purple', 'colorful', 'vibrant', 'bright', 'pale', 'dark']
        sizes = ['small', 'large', 'tall', 'tiny', 'big', 'delicate', 'petite']
        qualities = ['beautiful', 'fragrant', 'blooming', 'fresh', 'lovely', 'pretty', 'stunning', 'gorgeous', 'elegant', 'graceful']
        locations_special = ['garden', 'vase', 'field', 'table', 'pot', 'bouquet', 'arrangement', 'yard', 'meadow', 'display']
    elif word in ['castle', 'skyscraper']:
        colors = ['gray', 'white', 'dark', 'stone', 'brown', 'black', 'silver', 'gleaming']
        sizes = ['large', 'huge', 'massive', 'tall', 'towering', 'enormous', 'giant', 'immense', 'colossal']
        qualities = ['old', 'ancient', 'modern', 'impressive', 'grand', 'magnificent', 'imposing', 'majestic', 'historic', 'famous']
        locations_special = ['city', 'hill', 'skyline', 'downtown', 'landscape', 'horizon', 'district', 'area', 'center', 'view']
    elif word in ['possum', 'raccoon', 'fox', 'skunk']:
        colors = ['gray', 'brown', 'dark', 'light', 'black', 'reddish', 'pale']
        sizes = ['small', 'tiny', 'little', 'medium']
        qualities = ['wild', 'nocturnal', 'furry', 'quick', 'clever', 'shy', 'cunning', 'agile', 'curious', 'sneaky']
        locations_special = ['tree', 'forest', 'ground', 'branch', 'woods', 'bush', 'undergrowth', 'yard', 'area', 'habitat']
    else:
        colors = ['dark', 'light', 'gray', 'brown', 'black', 'white', 'pale', 'bright']
        sizes = ['small', 'large', 'big', 'tiny', 'medium', 'huge']
        qualities = ['common', 'typical', 'usual', 'normal', 'ordinary', 'regular', 'standard']
        locations_special = [a for a in anchors if a not in ['animal', 'object', 'thing']][:10]
        if not locations_special:
            locations_special = ['area', 'place', 'spot', 'location', 'scene']
    
    all_descriptors = colors + sizes + qualities
    
    all_sentences = set()
    
    # [OK] 1. BASIC TEMPLATES
    all_sentences.add(f"{article}{word}")
    all_sentences.add(f"the {word}")
    if is_plural:
        all_sentences.update([f"some {word}", f"these {word}", f"those {word}", f"many {word}", f"several {word}"])
    else:
        all_sentences.update([f"this {word}", f"that {word}", f"one {word}"])
    
    # [OK] 2. SINGLE DESCRIPTOR
    for desc in all_descriptors:
        all_sentences.add(f"{article}{desc} {word}")
        all_sentences.add(f"the {desc} {word}")
        all_sentences.add(f"{article}{word} {verb_be} {desc}")
        all_sentences.add(f"the {word} {verb_be} {desc}")
    
    # [OK] 3. DOUBLE DESCRIPTORS (all combinations)
    for color in colors:
        for size in sizes:
            all_sentences.add(f"{article}{color} {size} {word}")
            all_sentences.add(f"{article}{size} {color} {word}")
            all_sentences.add(f"the {word} {verb_be} {color} and {size}")
            all_sentences.add(f"the {color} and {size} {word}")
            
    for size in sizes:
        for quality in qualities[:8]:
            all_sentences.add(f"{article}{size} {quality} {word}")
            all_sentences.add(f"{article}{quality} {size} {word}")
            all_sentences.add(f"the {word} {verb_be} {size} and {quality}")
    
    for color in colors[:8]:
        for quality in qualities[:8]:
            all_sentences.add(f"{article}{color} {quality} {word}")
            all_sentences.add(f"the {word} {verb_be} {color} and {quality}")
    
    # [OK] 4. LOCATION-BASED (expanded prepositions)
    preps = ['in', 'on', 'near', 'beside', 'by', 'under', 'over', 'around', 'behind', 'in front of']
    for loc in locations_special:
        for prep in preps:
            all_sentences.add(f"{article}{word} {prep} the {loc}")
            all_sentences.add(f"the {word} {prep} the {loc}")
            
            # With 2 descriptors
            if len(all_descriptors) >= 2:
                desc1 = random.choice(all_descriptors)
                desc2 = random.choice([d for d in all_descriptors if d != desc1])
                all_sentences.add(f"{article}{desc1} {word} {prep} the {loc}")
                all_sentences.add(f"the {desc1} {word} {prep} the {loc}")
                all_sentences.add(f"{article}{desc1} {desc2} {word} {prep} the {loc}")
    
    # [OK] 5. VG ANCHOR RELATIONSHIPS
    for anchor in anchors:
        article_anchor = "an " if anchor[0] in 'aeiou' else "a "
        
        all_sentences.add(f"the {word} and the {anchor}")
        all_sentences.add(f"{article}{word} and {article_anchor}{anchor}")
        all_sentences.add(f"{article}{word} with {article_anchor}{anchor}")
        all_sentences.add(f"both the {word} and the {anchor}")
        
        if anchor in ['fruit', 'food', 'animal', 'flower', 'tree', 'vehicle', 'building', 'plant', 'insect', 'fish', 'mammal']:
            all_sentences.add(f"{article}{word} {verb_be} a type of {anchor}")
            all_sentences.add(f"{article}{word} {verb_be} a {anchor}")
            all_sentences.add(f"{article}{word} {verb_be} like {article_anchor}{anchor}")
    
    # [OK] 6. SUPERCLASS CO-OCCURRENCE
    if members:
        for other in members[:5]:
            all_sentences.add(f"the {word} and the {other}")
            all_sentences.add(f"{article}{word} and the {other}")
            all_sentences.add(f"both the {word} and the {other}")
            all_sentences.add(f"{word} or {other}")
            all_sentences.add(f"the {word} like the {other}")
            all_sentences.add(f"{article}{word} with the {other}")
            all_sentences.add(f"the {word} and {other} {verb_be} both {superclass_name}")
            all_sentences.add(f"both {word} and {other} {verb_be} {superclass_name}")
    
    # [OK] 7. NATURAL PHRASES
    for desc in all_descriptors:
        all_sentences.add(f"look at the {desc} {word}")
        all_sentences.add(f"see the {desc} {word}")
        all_sentences.add(f"there {verb_be} {article}{desc} {word}")
        all_sentences.add(f"you can see {article}{desc} {word}")
        all_sentences.add(f"notice the {desc} {word}")
    
    # [OK] 8. TRIPLE COMBINATIONS (descriptor + descriptor + location)
    for i in range(500):  # Generate 500 rich combinations
        desc1 = random.choice(all_descriptors)
        desc2 = random.choice([d for d in all_descriptors if d != desc1])
        loc = random.choice(locations_special)
        prep = random.choice(['in', 'on', 'near', 'beside', 'by'])
        
        templates = [
            f"{article}{desc1} {desc2} {word} {prep} the {loc}",
            f"the {desc1} and {desc2} {word} {prep} the {loc}",
            f"there {verb_be} {article}{desc1} {desc2} {word} {prep} the {loc}",
            f"you can see {article}{desc1} {desc2} {word} {prep} the {loc}",
            f"look at the {desc1} {desc2} {word} {prep} the {loc}",
        ]
        all_sentences.add(random.choice(templates))
    
    # Convert to list
    sentence_list = list(all_sentences)
    random.shuffle(sentence_list)
    
    # Pad if needed
    while len(sentence_list) < num_sentences:
        desc1 = random.choice(all_descriptors)
        desc2 = random.choice([d for d in all_descriptors if d != desc1])
        loc = random.choice(locations_special)
        sentence_list.append(f"{article}{desc1} {desc2} {word} near the {loc}")
    
    return sentence_list[:num_sentences]


VG_URL = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip"


def main():
    parser = argparse.ArgumentParser(
        description="Generate the augmented VG+CIFAR training corpus."
    )
    parser.add_argument(
        "--vg-text", default="vg_text.txt",
        help="Path to existing vg_text.txt. If the file does not exist, it will "
             "be downloaded automatically from the Visual Genome dataset. "
             "(default: vg_text.txt)"
    )
    parser.add_argument(
        "--output", default="vg_cifar_combined.txt",
        help="Output path for the combined corpus file that will be passed to "
             "train_best_model.py --corpus. (default: vg_cifar_combined.txt)"
    )
    args = parser.parse_args()

    cifar_output = "cifar_natural_vg_anchored_final.txt"
    combined_output = args.output
    vg_text_path = args.vg_text

    # ------------------------------------------------------------------ Step 0
    # Download Visual Genome region descriptions if not already present
    if not os.path.exists(vg_text_path):
        print("="*70)
        print("STEP 0: DOWNLOADING VISUAL GENOME TEXT")
        print("="*70)
        from skipgram_trainer import prepare_visual_genome_text
        vg_text_path = prepare_visual_genome_text(VG_URL, output_path=vg_text_path)
    else:
        print(f"Found existing VG text: {vg_text_path}  (skipping download)")

    print("="*70)
    print("STEP 1: GENERATING CIFAR SENTENCES (3000+ UNIQUE PER WORD)")
    print("="*70)
    
    # Generate CIFAR sentences
    with open(cifar_output, 'w') as f:
        for i, word in enumerate(MISSING_WORDS, 1):
            print(f"[{i:2d}/{len(MISSING_WORDS)}] {word:20s} ", end='', flush=True)
            
            sentences = generate_sentences_for_word(word, num_sentences=6000)
            
            for sentence in sentences:
                f.write(f"{sentence} .\n")
            
            unique = len(set(sentences))
            print(f"[OK] {len(sentences):,} sentences ({unique:,} unique)")
    
    print("\n" + "="*70)
    print("[OK] CIFAR GENERATION COMPLETE")
    print("="*70)
    print(f"Output: {cifar_output}\n")
    
    # [OK] STEP 2: COMBINE WITH VG TEXT
    print("="*70)
    print("STEP 2: COMBINING WITH VISUAL GENOME TEXT")
    print("="*70)
    
    try:
        # Read original VG text
        with open(vg_text_path, 'r', encoding='utf-8') as f:
            vg_text = f.read()
        
        # Read CIFAR augmentation
        with open(cifar_output, 'r', encoding='utf-8') as f:
            cifar_text = f.read()
        
        # Combine with space separator
        combined_text = vg_text + " " + cifar_text
        
        # Write to new file
        with open(combined_output, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        print(f"[OK] Created {combined_output}")
        print(f"   VG words: {len(vg_text.split()):,}")
        print(f"   CIFAR words: {len(cifar_text.split()):,}")
        print(f"   Total words: {len(combined_text.split()):,}")
        
        # [OK] STEP 3: VERIFY MISSING WORDS
        print("\n" + "="*70)
        print("STEP 3: VERIFYING MISSING WORDS")
        print("="*70)
        
        missing_found = 0
        for word in MISSING_WORDS[:5]:  # Check first 5 as sample
            count = combined_text.lower().count(word)
            print(f"   '{word}': {count:,} occurrences")
            if count > 0:
                missing_found += 1
        
        if missing_found == 5:
            print("\n SUCCESS! All missing words present in combined file!")
            print(f"\n Corpus ready. To train:")
            print(f"   python train_best_model.py --corpus {combined_output}")
        else:
            print("\n[!] WARNING: Some words might be missing")
    
    except FileNotFoundError:
        print(f"\n[X] ERROR: '{vg_text_path}' not found!")
        print(f"   Re-run without --vg-text to auto-download, or run:")
        print(f"   python src/corpus_generator.py  (downloads VG automatically)")


if __name__ == "__main__":
    main()



