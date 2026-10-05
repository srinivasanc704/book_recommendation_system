import os
import pandas as pd

def generate_books_dataset():
    """
    Creates a realistic dataset of 60 books across 15 distinct genres
    with detailed metadata and descriptions suitable for TF-IDF content-based filtering.
    """
    books = [
        # --- Fantasy ---
        {
            "book_id": 1,
            "title": "The Hobbit",
            "author": "J.R.R. Tolkien",
            "genre": "Fantasy",
            "description": "Bilbo Baggins, a peaceful hobbit, is whisked away on an unexpected journey across Middle-earth with a wizard and dwarves to reclaim a lost kingdom and treasure guarded by the dragon Smaug.",
            "rating": 4.8,
            "year": 1937
        },
        {
            "book_id": 2,
            "title": "The Fellowship of the Ring",
            "author": "J.R.R. Tolkien",
            "genre": "Fantasy",
            "description": "Frodo Baggins inherits a powerful magical ring and embarks on an epic quest across Middle-earth to destroy the One Ring in Mount Doom, battling dark lords, orcs, and treacherous creatures.",
            "rating": 4.9,
            "year": 1954
        },
        {
            "book_id": 3,
            "title": "Harry Potter and the Sorcerer's Stone",
            "author": "J.K. Rowling",
            "genre": "Fantasy",
            "description": "An orphaned young boy discovers he is a wizard on his eleventh birthday and enrolls in Hogwarts School of Witchcraft and Wizardry, encountering magic, spells, friendships, and dark wizards.",
            "rating": 4.7,
            "year": 1997
        },
        {
            "book_id": 4,
            "title": "A Game of Thrones",
            "author": "George R.R. Martin",
            "genre": "Fantasy",
            "description": "In the kingdom of Westeros, noble families fight for control of the Iron Throne amidst political intrigue, dark magic, dragons, winter threats, and treacherous warfare.",
            "rating": 4.6,
            "year": 1996
        },
        {
            "book_id": 5,
            "title": "The Name of the Wind",
            "author": "Patrick Rothfuss",
            "genre": "Fantasy",
            "description": "The heroic life story of Kvothe, an infamous magician, musician, and scholar, recounting his adventures, arcane university education, and pursuit of legendary mythical beings.",
            "rating": 4.5,
            "year": 2007
        },

        # --- Science Fiction ---
        {
            "book_id": 6,
            "title": "1984",
            "author": "George Orwell",
            "genre": "Science Fiction",
            "description": "A chilling dystopian story about totalitarian rule, continuous surveillance by Big Brother, mind control, loss of freedom, and rebellion in a futuristic police state.",
            "rating": 4.7,
            "year": 1949
        },
        {
            "book_id": 7,
            "title": "Brave New World",
            "author": "Aldous Huxley",
            "genre": "Science Fiction",
            "description": "A futuristic dystopian society where citizens are genetically engineered, conditioned from birth, and pacified by technology and mind-altering drugs at the expense of genuine emotion.",
            "rating": 4.4,
            "year": 1932
        },
        {
            "book_id": 8,
            "title": "Dune",
            "author": "Frank Herbert",
            "genre": "Science Fiction",
            "description": "Set on the desert planet Arrakis, young Paul Atreides must navigate politics, interstellar empire conflicts, giant sandworms, and control of the universe's most precious spice resource.",
            "rating": 4.7,
            "year": 1965
        },
        {
            "book_id": 9,
            "title": "Neuromancer",
            "author": "William Gibson",
            "genre": "Science Fiction",
            "description": "A seminal cyberpunk sci-fi thriller following Case, a washed-up computer hacker hired by a mysterious employer to pull off the ultimate cyber heist against a powerful artificial intelligence.",
            "rating": 4.3,
            "year": 1984
        },
        {
            "book_id": 10,
            "title": "Foundation",
            "author": "Isaac Asimov",
            "genre": "Science Fiction",
            "description": "Psychohistorian Hari Seldon foresees the inevitable collapse of the Galactic Empire and creates a foundation of scientists to preserve human knowledge and rebuild civilization across galaxies.",
            "rating": 4.5,
            "year": 1951
        },

        # --- Mystery ---
        {
            "book_id": 11,
            "title": "The Hound of the Baskervilles",
            "author": "Arthur Conan Doyle",
            "genre": "Mystery",
            "description": "Brilliant detective Sherlock Holmes and Dr. Watson investigate the legendary curse of a demonic, supernatural hound terrorizing an aristocratic family on the eerie Devon moors.",
            "rating": 4.6,
            "year": 1902
        },
        {
            "book_id": 12,
            "title": "And Then There Were None",
            "author": "Agatha Christie",
            "genre": "Mystery",
            "description": "Ten strangers are lured to an isolated island mansion by an unknown host, only to be accused of past crimes and murdered one by one according to a sinister nursery rhyme.",
            "rating": 4.8,
            "year": 1939
        },
        {
            "book_id": 13,
            "title": "The Murder of Roger Ackroyd",
            "author": "Agatha Christie",
            "genre": "Mystery",
            "description": "Famous Belgian detective Hercule Poirot investigates the cunning murder of a wealthy country squire, uncovering buried secrets, blackmail, and an astonishing twist ending.",
            "rating": 4.7,
            "year": 1926
        },
        {
            "book_id": 14,
            "title": "The Girl with the Dragon Tattoo",
            "author": "Stieg Larsson",
            "genre": "Mystery",
            "description": "Disgraced journalist Mikael Blomkvist and genius hacker Lisbeth Salander team up to investigate the mysterious 40-year-old disappearance of a wealthy industrialist's niece.",
            "rating": 4.4,
            "year": 2005
        },

        # --- Thriller ---
        {
            "book_id": 15,
            "title": "Gone Girl",
            "author": "Gillian Flynn",
            "genre": "Thriller",
            "description": "On their fifth wedding anniversary, Amy Dunne suddenly vanishes, leaving behind signs of violence and suspicion pointing squarely toward her husband Nick in a maze of deception.",
            "rating": 4.3,
            "year": 2012
        },
        {
            "book_id": 16,
            "title": "The Da Vinci Code",
            "author": "Dan Brown",
            "genre": "Thriller",
            "description": "Symbologist Robert Langdon investigates a murder at the Louvre Museum, uncovering hidden symbols, secret societies, religious conspiracies, and ancient historical secrets.",
            "rating": 4.2,
            "year": 2003
        },
        {
            "book_id": 17,
            "title": "The Silent Patient",
            "author": "Alex Michaelides",
            "genre": "Thriller",
            "description": "Alicia Berenson shoots her husband five times in the face and never speaks another word; a psychotherapist becomes obsessed with uncovering her motive and dark psychological secrets.",
            "rating": 4.5,
            "year": 2019
        },
        {
            "book_id": 18,
            "title": "Angels and Demons",
            "author": "Dan Brown",
            "genre": "Thriller",
            "description": "Robert Langdon races through Rome and the Vatican to stop a secret brotherhood from using antimatter weapons to destroy the Catholic Church during a papal conclave.",
            "rating": 4.3,
            "year": 2000
        },

        # --- Fiction ---
        {
            "book_id": 19,
            "title": "To Kill a Mockingbird",
            "author": "Harper Lee",
            "genre": "Fiction",
            "description": "In the racially divided American South, lawyer Atticus Finch defends a wrongly accused black man, seen through the innocent eyes of his young daughter Scout.",
            "rating": 4.8,
            "year": 1960
        },
        {
            "book_id": 20,
            "title": "The Great Gatsby",
            "author": "F. Scott Fitzgerald",
            "genre": "Fiction",
            "description": "Narrator Nick Carraway observes the glamorous lifestyle, romantic obsession, and tragic downfall of millionaire Jay Gatsby in the roaring twenties of Long Island.",
            "rating": 4.4,
            "year": 1925
        },
        {
            "book_id": 21,
            "title": "The Catcher in the Rye",
            "author": "J.D. Salinger",
            "genre": "Fiction",
            "description": "Teenager Holden Caulfield wanders New York City after being expelled from prep school, wrestling with alienation, teenage angst, phoniness, and growing up.",
            "rating": 4.0,
            "year": 1951
        },
        {
            "book_id": 22,
            "title": "The Alchemist",
            "author": "Paulo Coelho",
            "genre": "Fiction",
            "description": "A mystical philosophical fable about Santiago, an Andalusian shepherd boy who journeys across the desert to the Egyptian pyramids in search of treasure and his true destiny.",
            "rating": 4.6,
            "year": 1988
        },

        # --- Romance ---
        {
            "book_id": 23,
            "title": "Pride and Prejudice",
            "author": "Jane Austen",
            "genre": "Romance",
            "description": "Spirited Elizabeth Bennet and proud aristocratic Mr. Darcy overcome societal expectations, class prejudice, and initial misunderstandings to discover true love.",
            "rating": 4.7,
            "year": 1813
        },
        {
            "book_id": 24,
            "title": "Jane Eyre",
            "author": "Charlotte Bronte",
            "genre": "Romance",
            "description": "An orphaned governess Jane Eyre falls in love with the brooding, enigmatic Mr. Rochester at Thornfield Hall, uncovering a dark secret concealed in the attic.",
            "rating": 4.5,
            "year": 1847
        },
        {
            "book_id": 25,
            "title": "The Fault in Our Stars",
            "author": "John Green",
            "genre": "Romance",
            "description": "Two teenage cancer patients Hazel and Augustus meet in a support group, embarking on a poignant, humorous, and heartbreaking love journey to Amsterdam.",
            "rating": 4.3,
            "year": 2012
        },
        {
            "book_id": 26,
            "title": "The Notebook",
            "author": "Nicholas Sparks",
            "genre": "Romance",
            "description": "A timeless passionate love story between Noah Calhoun and Allie Nelson, separated by social class and war, remembered through the pages of an old notebook.",
            "rating": 4.2,
            "year": 1996
        },

        # --- Adventure ---
        {
            "book_id": 27,
            "title": "Treasure Island",
            "author": "Robert Louis Stevenson",
            "genre": "Adventure",
            "description": "Young Jim Hawkins embarks on a perilous high-seas sailing voyage in search of buried pirate treasure, facing mutinous buccaneers and the cunning Long John Silver.",
            "rating": 4.5,
            "year": 1883
        },
        {
            "book_id": 28,
            "title": "Life of Pi",
            "author": "Yann Martel",
            "genre": "Adventure",
            "description": "After a devastating shipwreck in the Pacific Ocean, young Indian boy Pi Patel survives for 227 days stranded on a lifeboat alongside a fearsome Bengal tiger named Richard Parker.",
            "rating": 4.4,
            "year": 2001
        },
        {
            "book_id": 29,
            "title": "The Call of the Wild",
            "author": "Jack London",
            "genre": "Adventure",
            "description": "Buck, a domesticated dog stolen from his sunny California home, must survive the brutal conditions, harsh sled trails, and wilderness during the 1890s Klondike Gold Rush.",
            "rating": 4.3,
            "year": 1903
        },

        # --- Historical Fiction ---
        {
            "book_id": 30,
            "title": "The Book Thief",
            "author": "Markus Zusak",
            "genre": "Historical Fiction",
            "description": "Narrated by Death, young Liesel Meminger steals books and shares them with her foster family and a hidden Jewish fist-fighter in Nazi Germany during World War II.",
            "rating": 4.7,
            "year": 2005
        },
        {
            "book_id": 31,
            "title": "All the Light We Cannot See",
            "author": "Anthony Doerr",
            "genre": "Historical Fiction",
            "description": "A blind French girl and a gifted German orphan boy cross paths in occupied Saint-Malo, France, trying to survive the physical devastation and moral turmoil of World War II.",
            "rating": 4.6,
            "year": 2014
        },
        {
            "book_id": 32,
            "title": "The Kite Runner",
            "author": "Khaled Hosseini",
            "genre": "Historical Fiction",
            "description": "Amir seeks redemption for past betrayal of his childhood friend Hassan in war-torn Afghanistan, tracing decades of political upheaval, guilt, and family loyalty.",
            "rating": 4.7,
            "year": 2003
        },

        # --- Self-Help ---
        {
            "book_id": 33,
            "title": "Atomic Habits",
            "author": "James Clear",
            "genre": "Self-Help",
            "description": "A practical comprehensive framework for building good habits, breaking bad ones, and mastering the tiny continuous behaviors that lead to remarkable personal success.",
            "rating": 4.9,
            "year": 2018
        },
        {
            "book_id": 34,
            "title": "The 7 Habits of Highly Effective People",
            "author": "Stephen R. Covey",
            "genre": "Self-Help",
            "description": "A principle-centered guide for personal productivity, leadership, professional growth, character building, time management, and proactive decision making.",
            "rating": 4.7,
            "year": 1989
        },
        {
            "book_id": 35,
            "title": "Deep Work",
            "author": "Cal Newport",
            "genre": "Self-Help",
            "description": "Rules for focused success in a distracted world, teaching how to cultivate deep concentration, master cognitive skills, eliminate distractions, and boost output.",
            "rating": 4.6,
            "year": 2016
        },
        {
            "book_id": 36,
            "title": "The Subtle Art of Not Giving a F*ck",
            "author": "Mark Manson",
            "genre": "Self-Help",
            "description": "A refreshingly blunt approach to living a good life by accepting human limitations, embracing struggle, and choosing what truly matters instead of constant positivity.",
            "rating": 4.3,
            "year": 2016
        },

        # --- Psychology ---
        {
            "book_id": 37,
            "title": "Thinking, Fast and Slow",
            "author": "Daniel Kahneman",
            "genre": "Psychology",
            "description": "Nobel laureate Daniel Kahneman explains the two systems driving human thought: fast intuitive emotional thinking versus slow deliberate logical reasoning and cognitive biases.",
            "rating": 4.6,
            "year": 2011
        },
        {
            "book_id": 38,
            "title": "Quiet: The Power of Introverts",
            "author": "Susan Cain",
            "genre": "Psychology",
            "description": "An insightful exploration of how introverted people think, work, and thrive in an extrovert-dominated culture, highlighting the hidden strengths of quiet personalities.",
            "rating": 4.5,
            "year": 2012
        },
        {
            "book_id": 39,
            "title": "Man's Search for Meaning",
            "author": "Viktor E. Frankl",
            "genre": "Psychology",
            "description": "Psychiatrist Viktor Frankl chronicles his survival in Nazi concentration camps and introduces logotherapy, emphasizing finding purposeful meaning through human suffering.",
            "rating": 4.8,
            "year": 1946
        },
        {
            "book_id": 40,
            "title": "Influence: The Psychology of Persuasion",
            "author": "Robert B. Cialdini",
            "genre": "Psychology",
            "description": "Examines the key psychological principles of persuasion, social proof, reciprocation, authority, liking, and scarcity that lead people to say yes in everyday life.",
            "rating": 4.6,
            "year": 1984
        },

        # --- Business ---
        {
            "book_id": 41,
            "title": "The Lean Startup",
            "author": "Eric Ries",
            "genre": "Business",
            "description": "How modern entrepreneurs use continuous innovation, validated learning, and rapid experimentation to build successful businesses and sustainable startup products.",
            "rating": 4.5,
            "year": 2011
        },
        {
            "book_id": 42,
            "title": "Zero to One",
            "author": "Peter Thiel",
            "genre": "Business",
            "description": "Notes on startups and how to build the future by creating innovative monopolies from zero to one rather than copying existing competitive ideas.",
            "rating": 4.5,
            "year": 2014
        },
        {
            "book_id": 43,
            "title": "Good to Great",
            "author": "Jim Collins",
            "genre": "Business",
            "description": "A management study uncovering the disciplined leadership, culture, and core strategies that enable good companies to achieve sustained greatness over decades.",
            "rating": 4.6,
            "year": 2001
        },
        {
            "book_id": 44,
            "title": "Rich Dad Poor Dad",
            "author": "Robert T. Kiyosaki",
            "genre": "Business",
            "description": "Personal finance wisdom contrasting two father figures, teaching financial literacy, investing, assets versus liabilities, and building wealth independence.",
            "rating": 4.4,
            "year": 1997
        },

        # --- Technology ---
        {
            "book_id": 45,
            "title": "Clean Code",
            "author": "Robert C. Martin",
            "genre": "Technology",
            "description": "A handbook of agile software craftsmanship teaching software developers best practices, readable design, refactoring techniques, and writing maintainable code.",
            "rating": 4.7,
            "year": 2008
        },
        {
            "book_id": 46,
            "title": "The Pragmatic Programmer",
            "author": "Andrew Hunt, David Thomas",
            "genre": "Technology",
            "description": "Timeless practical career and coding advice for modern software engineers, covering software architecture, developer tools, debugging, and continuous learning.",
            "rating": 4.8,
            "year": 1999
        },
        {
            "book_id": 47,
            "title": "Artificial Intelligence: A Modern Approach",
            "author": "Stuart Russell, Peter Norvig",
            "genre": "Technology",
            "description": "The definitive comprehensive university textbook covering intelligent agents, machine learning algorithms, robotics, neural networks, and modern AI concepts.",
            "rating": 4.7,
            "year": 1995
        },
        {
            "book_id": 48,
            "title": "Algorithms to Live By",
            "author": "Brian Christian, Tom Griffiths",
            "genre": "Technology",
            "description": "How computer science algorithms, sorting, caching, probability, and decision-making principles can be applied to solve everyday human problems and time management.",
            "rating": 4.5,
            "year": 2016
        },

        # --- Biography ---
        {
            "book_id": 49,
            "title": "Steve Jobs",
            "author": "Walter Isaacson",
            "genre": "Biography",
            "description": "The riveting definitive biography of Apple co-founder Steve Jobs, chronicling his creative genius, relentless perfectionism, tech revolutions, and flawed personality.",
            "rating": 4.6,
            "year": 2011
        },
        {
            "book_id": 50,
            "title": "Elon Musk",
            "author": "Walter Isaacson",
            "genre": "Biography",
            "description": "An intimate inside look into the ambitious, tumultuous life of tech entrepreneur Elon Musk, driving revolutions in electric vehicles, space travel, and artificial intelligence.",
            "rating": 4.5,
            "year": 2023
        },
        {
            "book_id": 51,
            "title": "The Diary of a Young Girl",
            "author": "Anne Frank",
            "genre": "Biography",
            "description": "The poignant authentic diary of Jewish teenager Anne Frank hiding from Nazi persecution in a secret Amsterdam annex during World War II.",
            "rating": 4.8,
            "year": 1947
        },
        {
            "book_id": 52,
            "title": "Shoe Dog",
            "author": "Phil Knight",
            "genre": "Biography",
            "description": "A candid memoir by Nike founder Phil Knight detailing the harrowing early startup struggles, risks, perseverance, and triumph of building a global footwear brand.",
            "rating": 4.7,
            "year": 2016
        },

        # --- Philosophy ---
        {
            "book_id": 53,
            "title": "Meditations",
            "author": "Marcus Aurelius",
            "genre": "Philosophy",
            "description": "Private personal reflections of Roman Emperor Marcus Aurelius offering timeless Stoic philosophy on self-discipline, resilience, virtue, morality, and inner peace.",
            "rating": 4.7,
            "year": 180
        },
        {
            "book_id": 54,
            "title": "The Republic",
            "author": "Plato",
            "genre": "Philosophy",
            "description": "Plato's influential philosophical Socratic dialogue exploring justice, the ideal city-state society, philosopher kings, and the nature of human knowledge.",
            "rating": 4.3,
            "year": -375
        },
        {
            "book_id": 55,
            "title": "Letters from a Stoic",
            "author": "Seneca",
            "genre": "Philosophy",
            "description": "A collection of 124 moral letters from Seneca the Younger providing practical wisdom on living simply, overcoming fear of death, and cultivating inner tranquility.",
            "rating": 4.6,
            "year": 65
        },
        {
            "book_id": 56,
            "title": "Beyond Good and Evil",
            "author": "Friedrich Nietzsche",
            "genre": "Philosophy",
            "description": "Nietzsche critiques traditional Western morality, truth, religion, and dogmatic philosophy, advocating for independent free thinkers and will to power.",
            "rating": 4.2,
            "year": 1886
        },

        # --- Young Adult ---
        {
            "book_id": 57,
            "title": "The Hunger Games",
            "author": "Suzanne Collins",
            "genre": "Young Adult",
            "description": "In a post-apocalyptic nation of Panem, sixteen-year-old Katniss Everdeen volunteers to take her sister's place in a televised fight-to-the-death tournament.",
            "rating": 4.6,
            "year": 2008
        },
        {
            "book_id": 58,
            "title": "Divergent",
            "author": "Veronica Roth",
            "genre": "Young Adult",
            "description": "In a dystopian society divided into factions based on personality traits, Beatrice Prior discovers she is Divergent and uncovers a plot to destroy the faction system.",
            "rating": 4.2,
            "year": 2011
        },
        {
            "book_id": 59,
            "title": "The Maze Runner",
            "author": "James Dashner",
            "genre": "Young Adult",
            "description": "Thomas wakes up with no memories in a giant enclosed Glade surrounded by an ever-changing lethal stone maze filled with biomechanical monsters.",
            "rating": 4.3,
            "year": 2009
        },
        {
            "book_id": 60,
            "title": "Percy Jackson & the Olympians: The Lightning Thief",
            "author": "Rick Riordan",
            "genre": "Young Adult",
            "description": "Twelve-year-old Percy Jackson discovers he is the demigod son of Poseidon and is accused of stealing Zeus's master lightning bolt, launching an adventure across America.",
            "rating": 4.6,
            "year": 2005
        }
    ]

    df = pd.DataFrame(books)

    # Basic validations
    assert df["book_id"].nunique() == len(df), "Book IDs must be unique."
    assert df["title"].nunique() == len(df), "Book titles must be unique."
    assert not df.isnull().values.any(), "Dataset contains null values."

    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "books.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Successfully created {csv_path} with {len(df)} books across {df['genre'].nunique()} genres.")
    return df

if __name__ == "__main__":
    generate_books_dataset()
