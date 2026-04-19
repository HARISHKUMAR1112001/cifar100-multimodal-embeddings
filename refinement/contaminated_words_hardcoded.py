"""
Evaluation Settings and Contaminated Words
These words need GA refinement.
"""

# Words with 0/5 same-class neighbors (WORST)
CONTAMINATED_CRITICAL = [
    'aquariumfish',  # fish -> cans, cattle, whale, spider, sweetpeppers
    'bee',           # insects -> snake, turtle, crocodile, sweetpeppers, spider
    'beetle',        # insects -> seal, turtle, sweetpeppers, spider, snail
    'butterfly',     # insects -> seal, tractor, sweetpeppers, porcupine, lawnmower
    'camel',         # large_omnivores -> pickuptruck, beaver, seal, hamster, dolphin
    'caterpillar',   # insects -> sweetpeppers, spider, seal, ray, crab
    'cattle',        # large_omnivores -> squirrel, aquariumfish, ray, flatfish, sweetpeppers
    'cockroach',     # insects -> spider, sweetpeppers, worm, crab, beaver
    'computerkeyboard',  # household -> seal, lawnmower, crab, whale, pickuptruck
]

# Words with 1/5 same-class neighbors (SEVERE)
CONTAMINATED_SEVERE = [
    'beaver',        # aquatic_mammals -> lawnmower, snail, spider, seal
    'castle',        # large_outdoor -> skyscraper, crocodile, beaver, lawnmower
    'crab',          # invertebrates -> spider, sweetpeppers, ray, turtle
    'crocodile',     # reptiles -> bee, seal, computerkeyboard, dolphin
    'dolphin',       # aquatic_mammals -> hamster, beaver, turtle, dinosaur
]

# Words with 2/5 same-class neighbors (MODERATE)
CONTAMINATED_MODERATE = [
    'apples',        # fruits_vegetables -> oranges, pears, sunflowers
    'cans',          # food_containers -> bowls, aquariumfish, sweetpeppers
    'chimpanzee',    # large_omnivores -> kangaroo, sweetpeppers, lizard
]

# Combined list (prioritize critical first)
CONTAMINATED_WORDS_ALL = (
    CONTAMINATED_CRITICAL +
    CONTAMINATED_SEVERE +
    CONTAMINATED_MODERATE
)

# Class mapping for fitness function
WORD_TO_SUPERCLASS = {
    # aquatic_mammals
    'beaver': 'aquatic_mammals', 'dolphin': 'aquatic_mammals', 'otter': 'aquatic_mammals',
    'seal': 'aquatic_mammals', 'whale': 'aquatic_mammals',

    # fish
    'aquariumfish': 'fish', 'flatfish': 'fish', 'ray': 'fish', 'shark': 'fish', 'trout': 'fish',

    # insects
    'bee': 'insects', 'beetle': 'insects', 'butterfly': 'insects',
    'caterpillar': 'insects', 'cockroach': 'insects',

    # large_omnivores
    'camel': 'large_omnivores_herbivores', 'cattle': 'large_omnivores_herbivores',
    'chimpanzee': 'large_omnivores_herbivores', 'kangaroo': 'large_omnivores_herbivores',

    # food_containers
    'bottles': 'food_containers', 'bowls': 'food_containers', 'cans': 'food_containers',
    'cups': 'food_containers', 'plates': 'food_containers',

    # fruits_vegetables
    'apples': 'fruits_vegetables', 'mushrooms': 'fruits_vegetables',
    'oranges': 'fruits_vegetables', 'pears': 'fruits_vegetables',
    'sweetpeppers': 'fruits_vegetables',

    # invertebrates
    'crab': 'invertebrates', 'lobster': 'invertebrates', 'snail': 'invertebrates',
    'spider': 'invertebrates', 'worm': 'invertebrates',

    # reptiles
    'crocodile': 'reptiles', 'dinosaur': 'reptiles', 'lizard': 'reptiles',
    'snake': 'reptiles', 'turtle': 'reptiles',

    # household_items
    'computerkeyboard': 'household_items', 'telephone': 'household_items',
    'television': 'household_items', 'wardrobe': 'household_items',

    # large_outdoor_things
    'castle': 'large_outdoor_things', 'skyscraper': 'large_outdoor_things',
}

# Superclass members (for class-aware fitness)
SUPERCLASS_MEMBERS = {
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

if __name__ == "__main__":
    print(f"Total contaminated words: {len(CONTAMINATED_WORDS_ALL)}")
    print(f"  Critical (0/5): {len(CONTAMINATED_CRITICAL)}")
    print(f"  Severe (1/5):   {len(CONTAMINATED_SEVERE)}")
    print(f"  Moderate (2/5): {len(CONTAMINATED_MODERATE)}")