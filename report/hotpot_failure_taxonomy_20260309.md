# Hotpot Failure Taxonomy Report

## 1. Inputs

- details: `experiments\runs\stage2_hotpot_matrix\hotpot_retrieval_dense_sharded_20260308_132925\details.json`
- metrics: `experiments\runs\stage2_hotpot_matrix\hotpot_retrieval_dense_sharded_20260308_132925\metrics.json`
- dense manifest: `E:\rag-benchmark-indexes\wiki18_21m_dense_sharded\manifest.json`
- title BM25 manifest: `not provided`
- dense probe top-k: `300`
- title probe top-k: `50`

## 2. Main Class Counts

| Main Class | Count | Pct |
| --- | ---: | ---: |
| `no_gold_in_raw` | `55` | `0.2750` |
| `only_one_gold_in_raw` | `96` | `0.4800` |
| `both_gold_after_dedup_but_lost_after_rerank` | `2` | `0.0100` |
| `both_gold_in_final` | `47` | `0.2350` |

## 3. Subcategory Counts

| Subcategory | Count | Pct | Recommendation |
| --- | ---: | ---: | --- |
| `budget_limited` | `22` | `0.1100` | increase raw dense depth or add title-aware prefilter before truncating candidates |
| `embedding_confusion` | `0` | `0.0000` | add lexical title prior or hybrid title retrieval before reranking |
| `normalization_or_alias_suspect` | `33` | `0.1650` | tighten title normalization and alias handling before candidate evaluation |
| `query_formulation_gap` | `96` | `0.4800` | prioritize query rewrite or hotpot_decompose instead of retriever-only tuning |
| `rerank_loss` | `2` | `0.0100` | adjust reranker or title packing because both gold titles already survived raw retrieval |
| `resolved` | `47` | `0.2350` | keep as a control group and do not optimize specifically for these samples |

## 4. Top Blockers

1. `query_formulation_gap`: 96 (prioritize query rewrite or hotpot_decompose instead of retriever-only tuning)
2. `normalization_or_alias_suspect`: 33 (tighten title normalization and alias handling before candidate evaluation)
3. `budget_limited`: 22 (increase raw dense depth or add title-aware prefilter before truncating candidates)
4. `rerank_loss`: 2 (adjust reranker or title packing because both gold titles already survived raw retrieval)

## 5. Representative Examples

### `budget_limited`

- `dev_6` Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?
  - gold: ['Eenasul Fateh', 'Management consulting']
  - missing_gold: ['Eenasul Fateh']
  - dense_probe_hits: ['Eenasul Fateh']
  - sparse_probe_hits: []
  - alias_candidates: {'Eenasul Fateh': ['Eenasul Fateh']}
- `dev_12` What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective?
  - gold: ["Oh My God (Guns N' Roses song)", 'End of Days (film)']
  - missing_gold: ["Oh My God (Guns N' Roses song)", 'End of Days (film)']
  - dense_probe_hits: ["Oh My God (Guns N' Roses song)"]
  - sparse_probe_hits: []
  - alias_candidates: {"Oh My God (Guns N' Roses song)": ["Oh My God (Guns N' Roses song)"]}
- `dev_14` The football manager who recruited David Beckham managed Manchester United during what timeframe?
  - gold: ['1995–96 Manchester United F.C. season', 'Alex Ferguson']
  - missing_gold: ['1995–96 Manchester United F.C. season', 'Alex Ferguson']
  - dense_probe_hits: ['Alex Ferguson']
  - sparse_probe_hits: []
  - alias_candidates: {'1995–96 Manchester United F.C. season': ['Manchester United F.C.', 'Manchester United F.C.', 'Manchester United F.C.', 'Manchester United F.C.', 'F.C. United of Manchester'], 'Alex Ferguson': ['Alex Ferguson']}
- `dev_16` The Vermont Catamounts men's soccer team currently competes in a conference that was formerly known as what from 1988 to 1996?
  - gold: ["Vermont Catamounts men's soccer", 'America East Conference']
  - missing_gold: ["Vermont Catamounts men's soccer", 'America East Conference']
  - dense_probe_hits: ['America East Conference']
  - sparse_probe_hits: []
  - alias_candidates: {"Vermont Catamounts men's soccer": ["Vermont Catamounts men's ice hockey", "Vermont Catamounts men's basketball", 'Vermont Catamounts', "2018–19 Vermont Catamounts men's basketball team", "2017–18 Vermont Catamounts men's basketball team"], 'America East Conference': ["2017 America East Conference Women's Soccer Tournament", 'America East Conference']}
- `dev_52` What race track in the midwest hosts a 500 mile race eavery May?
  - gold: ['1957 Indianapolis 500', 'Indianapolis Motor Speedway']
  - missing_gold: ['1957 Indianapolis 500', 'Indianapolis Motor Speedway']
  - dense_probe_hits: ['Indianapolis Motor Speedway']
  - sparse_probe_hits: []
  - alias_candidates: {'1957 Indianapolis 500': ['Indianapolis 500', 'Indianapolis 500', 'Indianapolis 500', 'Indianapolis 500'], 'Indianapolis Motor Speedway': ['Indianapolis Motor Speedway race results', 'Indianapolis Motor Speedway', 'Indianapolis Motor Speedway race results', 'Indianapolis Motor Speedway', 'Indianapolis Motor Speedway race results']}
- `dev_60` What distinction is held by the former NBA player who was a member of the Charlotte Hornets during their 1992-93 season and was head coach for the WNBA team Charlotte Sting?
  - gold: ['1992–93 Charlotte Hornets season', 'Muggsy Bogues']
  - missing_gold: ['Muggsy Bogues']
  - dense_probe_hits: ['Muggsy Bogues']
  - sparse_probe_hits: []
  - alias_candidates: {'Muggsy Bogues': ['Muggsy Bogues', 'Muggsy Bogues', 'Muggsy Bogues', 'Muggsy Bogues']}
- `dev_61` What is the name of the executive producer of the film that has a score composed by Jerry Goldsmith?
  - gold: ['Alien (soundtrack)', 'Alien (film)']
  - missing_gold: ['Alien (film)']
  - dense_probe_hits: ['Alien (film)']
  - sparse_probe_hits: []
  - alias_candidates: {'Alien (film)': ['Alien (soundtrack)', 'Alien (soundtrack)', 'Alien (soundtrack)', 'Alien (film)', 'Alien (soundtrack)']}
- `dev_80` What is the county seat of the county where East Lempster, New Hampshire is located?
  - gold: ['East Lempster, New Hampshire', 'Sullivan County, New Hampshire']
  - missing_gold: ['East Lempster, New Hampshire', 'Sullivan County, New Hampshire']
  - dense_probe_hits: ['Sullivan County, New Hampshire']
  - sparse_probe_hits: []
  - alias_candidates: {'East Lempster, New Hampshire': ['Lempster, New Hampshire', 'East Hampshire', 'East Hampshire (UK Parliament constituency)', 'Lempster, New Hampshire', 'Lempster, New Hampshire'], 'Sullivan County, New Hampshire': ['New Hampshire', 'New Hampshire', 'New Hampshire', 'Sullivan County, New Hampshire', 'Sullivan County, New Hampshire']}
- `dev_93` Do the drinks Gibson and Zurracapote both contain gin?
  - gold: ['Gibson (cocktail)', 'Zurracapote']
  - missing_gold: ['Gibson (cocktail)', 'Zurracapote']
  - dense_probe_hits: ['Gibson (cocktail)']
  - sparse_probe_hits: []
  - alias_candidates: {'Gibson (cocktail)': ['Gibson (cocktail)', 'Gibson (cocktail)', 'Gibson (cocktail)']}
- `dev_101` David Huntsinger has worked with this gospel singer born in the month of July?
  - gold: ['David Huntsinger', 'Larnelle Harris']
  - missing_gold: ['Larnelle Harris']
  - dense_probe_hits: ['Larnelle Harris']
  - sparse_probe_hits: []
  - alias_candidates: {'Larnelle Harris': ['Larnelle Harris']}

### `normalization_or_alias_suspect`

- `dev_10` What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?
  - gold: ['Kansas Song', 'University of Kansas']
  - missing_gold: ['Kansas Song']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'Kansas Song': ['Kansas (Kansas album)', 'Kansas (Kansas album)', 'Kansas (Kansas album)', 'Kansas (band)', 'Kansas (Kansas album)']}
- `dev_15` Brown State Fishing Lake is in a country that has a population of how many inhabitants ?
  - gold: ['Brown State Fishing Lake', 'Brown County, Kansas']
  - missing_gold: ['Brown State Fishing Lake']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'Brown State Fishing Lake': ['Brown Lake (Stradbroke Island)', 'Brown Lake (Stradbroke Island)', 'Fishing Lake']}
- `dev_19` Which writer was from England, Henry Roth or Robert Erskine Childers?
  - gold: ['Henry Roth', 'Robert Erskine Childers']
  - missing_gold: ['Henry Roth', 'Robert Erskine Childers']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'Robert Erskine Childers': ['Erskine Childers (author)', 'Erskine Childers (author)', 'Erskine Childers (author)', 'Erskine Childers (author)', 'Erskine Childers (author)']}
- `dev_29` What is the name for the adventure in "Tunnels and Trolls", a game designed by Ken St. Andre?
  - gold: ['Arena of Khazan', 'Tunnels &amp; Trolls']
  - missing_gold: ['Arena of Khazan', 'Tunnels &amp; Trolls']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'Tunnels &amp; Trolls': ['Tunnels & Trolls', 'Tunnels (novel)', 'Tunnels & Trolls', 'Tunnels & Trolls', 'Tunnels & Trolls']}
- `dev_33` Are Freakonomics and In the Realm of the Hackers both American documentaries?
  - gold: ['Freakonomics (film)', 'In the Realm of the Hackers']
  - missing_gold: ['Freakonomics (film)']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'Freakonomics (film)': ['Freakonomics Radio', 'Freakonomics Radio', 'Freakonomics Radio']}
- `dev_36` Seven Brief Lessons on Physics was written by an Italian physicist that has worked in France since what year?
  - gold: ['Seven Brief Lessons on Physics', 'Carlo Rovelli']
  - missing_gold: ['Seven Brief Lessons on Physics']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'Seven Brief Lessons on Physics': ['Physics (Aristotle)', 'Physics (Aristotle)', 'Physics (Aristotle)']}
- `dev_39` Ralph Hefferline was a psychology professor at a university that is located in what city?
  - gold: ['Ralph Hefferline', 'Columbia University']
  - missing_gold: ['Columbia University']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'Columbia University': ['Teachers College, Columbia University']}
- `dev_48` Are Ferocactus and Silene both types of plant?
  - gold: ['Ferocactus', 'Silene']
  - missing_gold: ['Ferocactus', 'Silene']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'Ferocactus': ['Ferocactus hamatacanthus', 'Ferocactus viridescens', 'Ferocactus hamatacanthus', 'Ferocactus cylindraceus', 'Ferocactus hamatacanthus'], 'Silene': ['Silene verecunda', 'Silene invisa', 'Silene bernardina', 'Silene laciniata', 'Silene dioica']}
- `dev_50` Which year and which conference was the 14th season for this conference as part of the NCAA Division that the Colorado Buffaloes played in with a record of 2-6 in conference play?
  - gold: ['2009 Colorado Buffaloes football team', '2009 Big 12 Conference football season']
  - missing_gold: ['2009 Colorado Buffaloes football team', '2009 Big 12 Conference football season']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'2009 Colorado Buffaloes football team': ['1976 Colorado Buffaloes football team', '2011 Colorado Buffaloes football team', '2012 Colorado Buffaloes football team', '2013 Colorado Buffaloes football team', 'Colorado Buffaloes football']}
- `dev_56` D1NZ is a series based on what oversteering technique?
  - gold: ['D1NZ', 'Drifting (motorsport)']
  - missing_gold: ['D1NZ', 'Drifting (motorsport)']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'Drifting (motorsport)': ['Red Bull Drifting World Championship']}

### `query_formulation_gap`

- `dev_0` Were Scott Derrickson and Ed Wood of the same nationality?
  - gold: ['Scott Derrickson', 'Ed Wood']
  - missing_gold: ['Scott Derrickson']
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_1` What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
  - gold: ['Kiss and Tell (1945 film)', 'Shirley Temple']
  - missing_gold: ['Shirley Temple']
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_2` What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?
  - gold: ['The Hork-Bajir Chronicles', 'Animorphs']
  - missing_gold: ['The Hork-Bajir Chronicles', 'Animorphs']
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_5` 2014 S/S is the debut album of a South Korean boy group that was formed by who?
  - gold: ['2014 S/S', 'Winner (band)']
  - missing_gold: ['Winner (band)']
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_7` The arena where the Lewiston Maineiacs played their home games can seat how many people?
  - gold: ['Lewiston Maineiacs', 'Androscoggin Bank Colisée']
  - missing_gold: ['Androscoggin Bank Colisée']
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_9` Are Local H and For Against both from the United States?
  - gold: ['Local H', 'For Against']
  - missing_gold: ['For Against']
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_11` What screenwriter with credits for "Evolution" co-wrote a film starring Nicolas Cage and Téa Leoni?
  - gold: ['David Weissman', 'The Family Man']
  - missing_gold: ['The Family Man']
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_18` Roger O. Egeberg was Assistant Secretary for Health and Scientific Affairs during the administration of a president that served during what years?
  - gold: ['Roger O. Egeberg', 'Richard Nixon']
  - missing_gold: ['Richard Nixon']
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_20` Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990?
  - gold: ['Formula One drivers from Mexico', 'Sergio Pérez']
  - missing_gold: ['Sergio Pérez']
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_21` This singer of A Rather Blustery Day also voiced what hedgehog?
  - gold: ['A Rather Blustery Day', 'Jim Cummings']
  - missing_gold: ['A Rather Blustery Day']
  - dense_probe_hits: []
  - sparse_probe_hits: []

### `rerank_loss`

- `dev_84` What was the name of a woman from the book titled "Their Lives: The Women Targeted by the Clinton Machine " and was also a former white house intern?
  - gold: ['Their Lives', 'Monica Lewinsky']
  - missing_gold: []
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_161` Which man who presented the Australia 2022 FIFA World Cup bid was born on October 22, 1930?
  - gold: ['Australia 2022 FIFA World Cup bid', 'Frank Lowy']
  - missing_gold: []
  - dense_probe_hits: []
  - sparse_probe_hits: []

### `resolved`

- `dev_3` Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?
  - gold: ['Laleli Mosque', 'Esma Sultan Mansion']
  - missing_gold: []
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_4` The director of the romantic comedy "Big Stone Gap" is based in what New York city?
  - gold: ['Big Stone Gap (film)', 'Adriana Trigiani']
  - missing_gold: []
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_8` Who is older, Annie Morton or Terry Richardson?
  - gold: ['Annie Morton', 'Terry Richardson']
  - missing_gold: []
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_13` Are Random House Tower and 888 7th Avenue both used for real estate?
  - gold: ['Random House Tower', '888 7th Avenue']
  - missing_gold: []
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_17` Are Giuseppe Verdi and Ambroise Thomas both Opera composers ?
  - gold: ['Giuseppe Verdi', 'Ambroise Thomas']
  - missing_gold: []
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_24` What was the father of Kasper Schmeichel voted to be by the IFFHS in 1992?
  - gold: ['Kasper Schmeichel', 'Peter Schmeichel']
  - missing_gold: []
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_26` The 2011–12 VCU Rams men's basketball team, led by third year head coach Shaka Smart, represented Virginia Commonwealth University which was founded in what year?
  - gold: ["2011–12 VCU Rams men's basketball team", 'Virginia Commonwealth University']
  - missing_gold: []
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_32` Which  French ace pilot and adventurer fly L'Oiseau Blanc
  - gold: ["L'Oiseau Blanc", 'Charles Nungesser']
  - missing_gold: []
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_34` Which band, Letters to Cleo or Screaming Trees, had more members?
  - gold: ['Letters to Cleo', 'Screaming Trees']
  - missing_gold: []
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_35` Alexander Kerensky was defeated and destroyed by the Bolsheviks in the course of a civil war that ended when ?
  - gold: ['Socialist Revolutionary Party', 'Russian Civil War']
  - missing_gold: []
  - dense_probe_hits: []
  - sparse_probe_hits: []
