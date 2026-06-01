# Well-known facts for specific numbers.
# Used as a fallback when the model fails to generate a valid clue.
#
# Structure: { number: [(clue_text, [domains]), ...] }
#   clue_text — works inline: "Add {clue_text}."
#   domains   — tags used to pick the clue that best matches the user's chosen domain.
#
# Domain vocabulary (keep consistent):
#   "sports", "mathematics", "science", "biology", "chemistry", "physics",
#   "music", "history", "literature", "technology", "geography", "culture",
#   "games", "film", "language", "politics", "astronomy", "food"

import numpy as np
from sentence_transformers import SentenceTransformer

_CANONICAL_TAGS = [
    "sports", "mathematics", "science", "biology", "chemistry", "physics",
    "music", "history", "literature", "technology", "geography", "culture",
    "games", "film", "language", "politics", "astronomy", "food",
]

_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
_tag_embeddings = _embed_model.encode(_CANONICAL_TAGS, normalize_embeddings=True)

_SIMILARITY_THRESHOLD = 0.45


def _normalize_domain(domain: str) -> str:
    """Map a free-text domain to the closest canonical tag via cosine similarity.
    Returns the original lowercased string if no tag clears the threshold.
    """
    vec = _embed_model.encode(domain.lower().strip(), normalize_embeddings=True)
    sims = _tag_embeddings @ vec
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= _SIMILARITY_THRESHOLD:
        return _CANONICAL_TAGS[best_idx]
    return domain.lower().strip()


def get_hardcoded_clue(number: int, domain: str) -> str | None:
    """Return the clue for `number` whose domains best match `domain`.

    The domain is first normalized to the closest canonical tag via semantic
    similarity, so typos, abbreviations, and synonyms are handled automatically.

    Returns None if:
      - the number has no entry, or
      - no clue's domain tags match the given domain at all.

    Scoring per clue (highest wins):
      2 — exact match (domain == tag)
      1 — partial match (domain contains tag, or tag contains domain)

    Among tied scores, the first clue in the list wins.
    """
    entries = HARDCODED_CLUES.get(number)
    if not entries:
        return None

    domain_lower = _normalize_domain(domain)

    def score(tags):
        best = 0
        for tag in tags:
            if domain_lower == tag:
                return 2
            if domain_lower in tag or tag in domain_lower:
                best = max(best, 1)
        return best

    best_clue, best_score = None, 0
    for clue_text, tags in entries:
        s = score(tags)
        if s > best_score:
            best_score = s
            best_clue = clue_text

    return best_clue

HARDCODED_CLUES = {
    1: [
        ("The number of Earths in the solar system",                                                  ["science", "astronomy"]),
        ("The number of gold medals awarded on the top step of the Olympic podium",                   ["sports"]),
        ("The number of the loneliest, according to the classic rock song by Three Dog Night",        ["music", "culture"]),
    ],
    2: [
        ("The number of eyes on a human face",                                                        ["biology"]),
        ("The number of players in a game of chess",                                                  ["games", "sports"]),
        ("The number of poles on planet Earth (North and South)",                                     ["geography", "science"]),
    ],
    3: [
        ("The number of sides of a triangle",                                                         ["mathematics"]),
        ("The number of medals awarded on an Olympic podium (gold, silver, bronze)",                  ["sports"]),
        ("The number of strikes needed to bowl a turkey in bowling",                                  ["sports"]),
    ],
    4: [
        ("The number of seasons in a year",                                                           ["science", "culture"]),
        ("The number of Grand Slam tennis tournaments played each year",                              ["sports"]),
        ("The number of strings on a violin",                                                         ["music"]),
    ],
    5: [
        ("The number of rings on the Olympic flag",                                                   ["sports"]),
        ("The number of fingers on one human hand",                                                   ["biology"]),
        ("The number of vowels in the English alphabet",                                              ["language"]),
    ],
    6: [
        ("The number of faces on a cube",                                                             ["mathematics"]),
        ("The number of strings on a standard guitar",                                                ["music"]),
        ("The number of sides on a honeycomb cell",                                                   ["biology", "science"]),
    ],
    7: [
        ("The number of days in a week",                                                              ["culture", "history"]),
        ("The number of notes in a musical scale (do re mi fa sol la ti)",                            ["music"]),
        ("The number of wonders of the ancient world",                                                ["history"]),
        ("The number of colors in a rainbow",                                                         ["science", "physics"]),
    ],
    8: [
        ("The number of planets in the solar system",                                                 ["science", "astronomy"]),
        ("The number of tentacles on an octopus",                                                     ["biology"]),
        ("The number of bits in a byte",                                                              ["technology"]),
    ],
    9: [
        ("The number of lives a cat is said to have",                                                 ["culture", "biology"]),
        ("The number of innings in a standard baseball game",                                         ["sports"]),
        ("The number of justices on the US Supreme Court",                                            ["politics", "history"]),
    ],
    10: [
        ("The number of fingers on a pair of human hands",                                            ["biology"]),
        ("The number of events in a decathlon",                                                       ["sports"]),
        ("The number of frames in a standard game of ten-pin bowling",                                ["sports", "games"]),
    ],
    11: [
        ("The number of players on a soccer team",                                                    ["sports"]),
        ("The number of players on a cricket team",                                                   ["sports"]),
        ("The number of players on a field hockey team",                                              ["sports"]),
    ],
    12: [
        ("The number of months in a year",                                                            ["culture", "science"]),
        ("The number of signs of the Western zodiac",                                                 ["culture", "astronomy"]),
        ("The number of semitones in a musical octave",                                               ["music"]),
    ],
    13: [
        ("The number considered unlucky in many Western cultures",                                    ["culture"]),
        ("The number of players on a rugby league team",                                              ["sports"]),
        ("The number of original colonies that declared independence from Britain",                   ["history", "politics"]),
    ],
    14: [
        ("The number of lines in a sonnet",                                                           ["literature"]),
        ("The number of days in a fortnight",                                                         ["language", "culture"]),
        ("The number of mountains above 8,000 meters in the world",                                  ["geography", "sports"]),
    ],
    15: [
        ("The number of players on a rugby union team",                                               ["sports"]),
        ("The number of billiard balls in a standard rack",                                           ["sports", "games"]),
        ("The number of red balls racked at the start of a snooker game",                             ["sports", "games"]),
    ],
    16: [
        ("The total number of pawns on a chessboard at the start of a game (8 per side)",             ["games"]),
        ("The number of ounces in a pound",                                                           ["science", "culture"]),
        ("The number of black squares on one half of a chessboard",                                   ["games", "mathematics"]),
    ],
    17: [
        ("The number of syllables in a haiku (5-7-5)",                                               ["literature", "culture"]),
        ("The number most frequently chosen when people are asked to pick a random number from 1–20", ["mathematics", "culture"]),
        ("The number of stripes on a regulation volleyball",                                          ["sports"]),
    ],
    18: [
        ("The number of holes on a standard golf course",                                             ["sports"]),
        ("The age at which a person reaches legal adulthood in many countries",                       ["culture", "politics"]),
        ("The number of wheels on a standard 18-wheel semi-truck",                                   ["technology", "culture"]),
    ],
    19: [
        ("The atomic number of potassium (K) in the periodic table",                                 ["science", "chemistry"]),
        ("The number of the century when the Industrial Revolution began (the 1800s)",                ["history", "technology"]),
        ("The number of the World Cup hosted by South Africa in 2010 (the 19th FIFA World Cup)",      ["sports", "history"]),
    ],
    20: [
        ("The number of fingers and toes on a human body",                                            ["biology"]),
        ("The number of questions in the classic guessing game '20 Questions'",                       ["games", "culture"]),
        ("The number of amino acids used to build proteins in living organisms",                      ["biology", "chemistry"]),
    ],
    21: [
        ("The number of dots on a standard six-sided die",                                            ["mathematics", "games"]),
        ("The legal drinking age in the United States",                                               ["culture", "politics"]),
        ("The target number in the card game Blackjack",                                              ["games", "culture"]),
    ],
    22: [
        ("The number of letters in the Hebrew alphabet",                                              ["language", "culture", "history"]),
        ("The number of players on the field at the start of a cricket match (11 per side)",          ["sports"]),
        ("The number in the title of Joseph Heller's satirical novel about military bureaucracy",     ["literature"]),
    ],
    23: [
        ("The number of pairs of chromosomes in the human body",                                      ["biology", "science"]),
        ("The jersey number Michael Jordan wore for most of his NBA career",                          ["sports"]),
        ("The smallest prime number greater than 20",                                                 ["mathematics"]),
    ],
    24: [
        ("The number of hours in a day",                                                              ["science", "culture"]),
        ("The number of frames per second in standard film",                                          ["film", "technology"]),
        ("The number of major and minor keys in Western tonal music (12 major + 12 minor)",           ["music"]),
    ],
    25: [
        ("The number of years in a silver anniversary",                                               ["culture"]),
        ("The number 5 squared — the perfect square between 20 and 30",                               ["mathematics"]),
        ("The number of cents in a US quarter dollar",                                                ["culture", "history"]),
    ],
    26: [
        ("The number of letters in the English alphabet",                                             ["language"]),
        ("The number of miles in a marathon (rounded down from 26.2)",                               ["sports"]),
        ("The number of bones in the human foot",                                                     ["biology"]),
    ],
    27: [
        ("The number of bones in the human hand",                                                     ["biology"]),
        ("The age at which an unusually high number of famous musicians died (the 27 Club)",          ["music", "culture"]),
        ("The number of amendments to the US Constitution",                                           ["history", "politics"]),
    ],
    28: [
        ("The number of days in February in a non-leap year",                                        ["science", "culture"]),
        ("The number of letters in the Arabic alphabet",                                              ["language", "culture"]),
        ("The number of dominoes in a standard double-six set",                                       ["games"]),
    ],
    29: [
        ("The atomic number of copper (Cu) in the periodic table",                                   ["science", "chemistry"]),
        ("The number of bones in the human skull",                                                    ["biology"]),
        ("The number of days in February during a leap year",                                         ["science", "culture"]),
    ],
    30: [
        ("The number of days in September, April, June, and November",                               ["culture", "science"]),
        ("The number of teams in the NBA",                                                            ["sports"]),
        ("The number of years in a pearl anniversary",                                                ["culture"]),
    ],
    31: [
        ("The number of days in January, March, May, July, August, October, and December",           ["culture", "science"]),
        ("The number of flavors Baskin-Robbins originally advertised",                               ["food", "culture"]),
        ("The prime number that follows 29",                                                          ["mathematics"]),
    ],
    32: [
        ("The number of teeth in a full adult human mouth",                                           ["biology"]),
        ("The temperature at which water freezes in degrees Fahrenheit",                             ["science", "physics"]),
        ("The number of teams in the NFL",                                                            ["sports"]),
    ],
    33: [
        ("The number of vertebrae in the human spine",                                                ["biology"]),
        ("The speed in rpm at which a standard vinyl LP record plays (33⅓)",                         ["music", "technology"]),
        ("The number of miners rescued in the 2010 Copiapó mining accident in Chile",                ["history"]),
    ],
    36: [
        ("The number of inches in a yard",                                                            ["science", "culture"]),
        ("The number of dramatic situations identified by French critic Georges Polti",               ["literature", "culture"]),
        ("The number of possible outcomes when rolling two standard dice (6×6)",                      ["mathematics", "games"]),
    ],
    37: [
        ("The normal human body temperature in degrees Celsius",                                      ["biology", "science"]),
        ("The number of plays attributed to William Shakespeare",                                     ["literature", "history"]),
        ("The atomic number of rubidium (Rb) in the periodic table",                                 ["science", "chemistry"]),
    ],
    40: [
        ("The number of weeks in a typical human pregnancy",                                          ["biology"]),
        ("The number of teeth in an adult horse's mouth",                                             ["biology"]),
        ("The number of thieves in the story Ali Baba and the Forty Thieves",                        ["literature", "culture"]),
    ],
    42: [
        ("The Answer to the Ultimate Question of Life, the Universe, and Everything, according to The Hitchhiker's Guide to the Galaxy", ["literature", "culture", "science fiction"]),
        ("The jersey number of Jackie Robinson, retired across all of Major League Baseball",         ["sports", "history"]),
        ("The angle in degrees at which a rainbow arc appears above the horizon",                     ["science", "physics"]),
    ],
    45: [
        ("The number of degrees in each base angle of an isosceles right triangle",                  ["mathematics"]),
        ("The speed in rpm at which a vinyl single (7-inch record) plays",                           ["music", "technology"]),
        ("The sum of all single-digit numbers (1+2+3+4+5+6+7+8+9)",                                 ["mathematics"]),
    ],
    46: [
        ("The number of chromosomes in a typical human cell",                                         ["biology", "science"]),
        ("The atomic number of palladium (Pd), used in catalytic converters",                        ["science", "chemistry", "technology"]),
        ("The year 46 BC, when Julius Caesar introduced the Julian calendar, standardizing 365 days per year", ["history"]),
    ],
    50: [
        ("The number of states in the United States of America",                                      ["geography", "politics", "history"]),
        ("The number of years in a golden anniversary",                                               ["culture"]),
        ("The frequency in hertz of standard alternating current (AC) in Europe and most of the world", ["science", "physics", "technology"]),
    ],
    52: [
        ("The number of cards in a standard deck without jokers",                                     ["games"]),
        ("The number of weeks in a year",                                                             ["culture", "science"]),
        ("The number of white keys on a standard piano",                                              ["music"]),
    ],
    55: [
        ("The Fibonacci number that follows 34 in the sequence",                                      ["mathematics"]),
        ("The atomic number of cesium (Cs), used in atomic clocks — the most precise timekeepers",   ["science", "chemistry", "technology"]),
        ("The number of delegates who attended the US Constitutional Convention of 1787",             ["history", "politics"]),
    ],
    57: [
        ("The number of varieties advertised in the Heinz slogan",                                    ["food", "culture"]),
        ("The atomic number of lanthanum (La), the first element in the lanthanide series",          ["science", "chemistry"]),
        ("The year 1957, when the Soviet Union launched Sputnik, the first artificial satellite",     ["history", "science", "technology", "astronomy"]),
    ],
    60: [
        ("The number of seconds in a minute",                                                         ["science", "culture"]),
        ("The atomic number of neodymium (Nd), used in the world's strongest permanent magnets",     ["science", "chemistry", "technology"]),
        ("The number of years in a diamond anniversary",                                              ["culture"]),
    ],
    64: [
        ("The number of squares on a chessboard",                                                     ["games"]),
        ("The number of the Nintendo 64 gaming console, released in 1996",                           ["technology", "culture"]),
        ("The number 2 raised to the power of 6 (2⁶)",                                               ["mathematics", "technology"]),
    ],
    66: [
        ("The atomic number of dysprosium (Dy) in the periodic table",                               ["science", "chemistry"]),
        ("The number of the famous American Route 66, stretching from Chicago to Los Angeles",       ["geography", "culture", "history"]),
        ("The UK state pension qualifying age as of 2024",                                            ["culture", "politics"]),
    ],
    72: [
        ("The typical par score for a full 18-hole round of golf",                                    ["sports"]),
        ("The number of hours in three days",                                                         ["science", "mathematics"]),
        ("The average human resting heart rate in beats per minute",                                  ["biology", "science"]),
    ],
    78: [
        ("The number of cards in a standard Tarot deck",                                              ["games", "culture"]),
        ("The speed in rpm at which early shellac records play",                                      ["music", "history", "technology"]),
        ("The atomic number of platinum (Pt), one of the rarest and most valuable metals on Earth",  ["science", "chemistry"]),
    ],
    79: [
        ("The atomic number of gold (Au) in the periodic table",                                      ["science", "chemistry"]),
        ("The year 1979, when Sony launched the Walkman and changed how we listen to music",          ["music", "technology", "history"]),
        ("The element directly before mercury (80) in the periodic table",                            ["science", "chemistry"]),
    ],
    80: [
        ("The number of days it took Phileas Fogg to travel around the world in Jules Verne's novel", ["literature"]),
        ("The number of years in an octogenarian's lifetime",                                         ["culture", "biology"]),
        ("The atomic number of mercury (Hg), the only metal that is liquid at room temperature",     ["science", "chemistry"]),
    ],
    81: [
        ("The number of squares on a standard 9×9 Sudoku grid",                                      ["games", "mathematics"]),
        ("The number 3 raised to the power of 4 (3⁴)",                                               ["mathematics"]),
        ("The number of chapters in the Tao Te Ching",                                               ["literature", "culture", "history"]),
    ],
    87: [
        ("The number evoked by Lincoln's phrase 'four score and seven years ago' in the Gettysburg Address", ["history", "politics"]),
        ("The octane rating of regular unleaded gasoline",                                            ["science", "technology"]),
        ("The atomic number of francium (Fr), the rarest naturally occurring element on Earth",      ["science", "chemistry"]),
    ],
    88: [
        ("The number of keys on a standard piano",                                                    ["music"]),
        ("The number of constellations recognized by the International Astronomical Union",           ["science", "astronomy"]),
        ("The speed in miles per hour the DeLorean needed to travel through time in Back to the Future", ["film", "culture"]),
    ],
    90: [
        ("The number of degrees in a right angle",                                                    ["mathematics", "geometry"]),
        ("The number of minutes in a standard soccer match",                                          ["sports"]),
        ("The atomic number of thorium (Th), a radioactive element considered a future nuclear fuel", ["science", "chemistry"]),
    ],
    92: [
        ("The atomic number of uranium (U), the heaviest naturally occurring element",               ["science", "chemistry"]),
        ("The number of naturally occurring chemical elements on Earth",                              ["science", "chemistry"]),
        ("The year 1992, when the Barcelona Olympics first allowed professional basketball players",  ["sports", "history"]),
    ],
    95: [
        ("The confidence level used as the standard threshold in most scientific studies (95% confidence interval)", ["science", "mathematics"]),
        ("The year Microsoft released Windows 95, transforming personal computing",                   ["technology", "history"]),
        ("The atomic number of americium (Am), the element used in household smoke detectors",       ["science", "chemistry"]),
    ],
    98: [
        ("The normal human body temperature in degrees Fahrenheit (98.6°F)",                         ["biology", "science"]),
        ("The atomic number of californium (Cf) in the periodic table",                              ["science", "chemistry"]),
        ("The year 1998, when Google was founded by Larry Page and Sergey Brin",                      ["technology", "history"]),
    ],
    99: [
        ("The number of bottles of beer on the wall in the classic counting song",                    ["culture", "music"]),
        ("The number of red balloons in the Nena song '99 Luftballons'",                              ["music"]),
        ("The jersey number of Wayne Gretzky, retired league-wide across the entire NHL",             ["sports"]),
    ],
    100: [
        ("The number of years in a century",                                                          ["history", "culture"]),
        ("The number of centimeters in a meter",                                                      ["science", "mathematics"]),
        ("The number of senators in the United States Senate",                                        ["politics", "history"]),
    ],
    101: [
        ("The number of Dalmatians in the Disney animated film",                                      ["film", "culture"]),
        ("The room number of the torture chamber in George Orwell's Nineteen Eighty-Four",           ["literature"]),
        ("The number used to designate an introductory university course",                            ["culture", "history"]),
    ],
    108: [
        ("The number of beads on a traditional Buddhist or Hindu prayer mala",                        ["culture", "history"]),
        ("The upper limit of the FM radio frequency band in MHz (87.5–108 MHz)",                     ["technology", "science"]),
        ("The number of stitches on a regulation Major League Baseball",                              ["sports"]),
    ],
    112: [
        ("The emergency phone number used across the European Union",                                 ["culture", "geography", "politics"]),
        ("The atomic number of copernicium (Cn), named after Nicolaus Copernicus",                   ["science", "chemistry", "history"]),
        ("The atomic mass of cadmium (Cd), used in rechargeable nickel-cadmium batteries",           ["science", "chemistry", "technology"]),
    ],
    118: [
        ("The number of confirmed elements in the periodic table",                                    ["science", "chemistry"]),
        ("The atomic number of oganesson (Og), the heaviest known element",                          ["science", "chemistry"]),
        ("The number of elements that have been officially named by IUPAC",                           ["science", "chemistry"]),
    ],
    128: [
        ("The number 2 raised to the power of 7 (2⁷)",                                               ["mathematics", "technology"]),
        ("The number of characters in the original ASCII character table",                            ["technology"]),
        ("The number of gigabytes in a common smartphone storage tier",                               ["technology"]),
    ],
    144: [
        ("The number of items in a gross (a dozen dozens)",                                           ["mathematics", "culture"]),
        ("The twelfth number in the Fibonacci sequence",                                              ["mathematics"]),
        ("The number of square inches in a square foot",                                              ["mathematics", "science"]),
    ],
    150: [
        ("The Pokédex number of Mewtwo, the final legendary Pokémon in Generation I",                ["culture", "games"]),
        ("The number of years in a sesquicentennial",                                                 ["history", "culture"]),
        ("The number of centimeters in one and a half meters",                                        ["science", "mathematics"]),
    ],
    180: [
        ("The number of degrees in the interior angles of a triangle",                               ["mathematics"]),
        ("The maximum score achievable in a single turn of darts (three triple-20s)",                ["sports", "games"]),
        ("The number of degrees in a straight angle",                                                 ["mathematics", "physics"]),
    ],
    206: [
        ("The number of bones in the adult human body",                                               ["biology"]),
        ("The number of National Olympic Committees recognized by the IOC",                           ["sports", "geography", "politics"]),
        ("The area code for Seattle, Washington",                                                     ["geography", "culture"]),
    ],
    360: [
        ("The number of degrees in a full circle",                                                    ["mathematics", "geometry"]),
        ("The name of Microsoft's Xbox gaming console released in 2005",                             ["technology", "culture"]),
        ("The number of degrees the Earth rotates in one full day",                                   ["science", "astronomy"]),
    ],
    365: [
        ("The number of days in a non-leap year",                                                     ["science", "culture"]),
        ("The approximate number of days it takes Earth to orbit the Sun",                           ["science", "astronomy"]),
        ("The sum of three consecutive perfect squares: 10² + 11² + 12² (100 + 121 + 144)",         ["mathematics"]),
    ],
    451: [
        ("The temperature in degrees Fahrenheit at which paper ignites, the title of Ray Bradbury's novel", ["literature", "science"]),
        ("The year 451 AD when Attila the Hun was defeated at the Battle of the Catalaunian Plains", ["history"]),
        ("The HTTP status code for 'Unavailable For Legal Reasons,' named after the Bradbury novel", ["technology", "literature"]),
    ],
    666: [
        ("The sum of all numbers on a standard European roulette wheel (0 through 36)",              ["mathematics", "games"]),
        ("The number considered ominous in many world cultures",                                      ["culture", "history"]),
        ("The value of the Roman numeral DCLXVI, which uses every major Roman numeral symbol exactly once (500+100+50+10+5+1)", ["mathematics", "history"]),
    ],
    1000: [
        ("The number of years in a millennium",                                                       ["history", "culture"]),
        ("The number of meters in a kilometer",                                                       ["science", "mathematics"]),
        ("The number of grams in a kilogram",                                                         ["science"]),
    ],
    1492: [
        ("The year Christopher Columbus first reached the Americas",                                  ["history", "geography"]),
        ("The year the last Emirate of Granada fell, ending 800 years of Moorish rule in Iberia",    ["history"]),
        ("The year the Ottoman Empire under Bayezid II welcomed Jews expelled from Spain",            ["history", "culture"]),
    ],
    1776: [
        ("The year the United States Declaration of Independence was signed",                         ["history", "politics"]),
        ("The year Adam Smith published The Wealth of Nations, founding modern economics",            ["history", "culture"]),
        ("The year Phi Beta Kappa, the oldest academic honor society in the United States, was founded", ["history", "culture"]),
    ],
    1969: [
        ("The year humans first walked on the Moon during the Apollo 11 mission",                    ["history", "science", "astronomy"]),
        ("The year of the Woodstock music festival in upstate New York",                              ["music", "history", "culture"]),
        ("The year the internet's predecessor, ARPANET, sent its first message",                     ["technology", "history"]),
    ],
    2000: [
        ("The year the world feared the Y2K bug would crash computers globally",                     ["technology", "history"]),
        ("The year of the Sydney Summer Olympic Games in Australia",                                  ["sports", "history"]),
        ("The year the Human Genome Project published its first working draft",                       ["science", "biology", "history"]),
    ],
    2001: [
        ("The year of the September 11 attacks, the deadliest terrorist attack in history",          ["history", "politics"]),
        ("The title of Stanley Kubrick's landmark science fiction film about space and evolution",    ["film", "science fiction"]),
        ("The year Wikipedia was founded, revolutionizing how the world shares knowledge",            ["technology", "history", "culture"]),
    ],
    2020: [
        ("The year the COVID-19 pandemic was declared a global emergency by the WHO",                ["history", "biology", "science"]),
        ("The year of the Tokyo Olympic Games, postponed by one year due to the pandemic",           ["sports", "history"]),
        ("The year SpaceX became the first private company to carry astronauts to the ISS",          ["science", "technology", "astronomy", "history"]),
    ],
    2024: [
        ("The year of the Paris Summer Olympic Games",                                                ["sports", "history"]),
        ("The year a total solar eclipse crossed North America from Mexico to Canada",               ["science", "astronomy"]),
        ("The year Argentina won their second consecutive Copa América title",                        ["sports", "history"]),
    ],
}
