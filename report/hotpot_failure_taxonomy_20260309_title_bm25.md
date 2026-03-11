# Hotpot Failure Taxonomy Report

## 1. Inputs

- details: `experiments\runs\hotpot_query_validation_stability\naive_baseline_dense_sharded_20260309_214927_rerank\hotpotqa\predictions.json`
- metrics: `experiments\runs\hotpot_query_validation_stability\naive_baseline_dense_sharded_20260309_214927_rerank\hotpotqa\metrics.json`
- dense manifest: `E:\rag-benchmark-indexes\wiki18_21m_dense_sharded\manifest.json`
- title BM25 manifest: `E:\rag-benchmark-indexes\wiki18_21m_title_bm25\manifest.json`
- dense probe top-k: `300`
- title probe top-k: `50`

## 2. Main Class Counts

| Main Class | Count | Pct |
| --- | ---: | ---: |
| `no_gold_in_raw` | `59` | `0.2950` |
| `only_one_gold_in_raw` | `91` | `0.4550` |
| `both_gold_after_dedup_but_lost_after_rerank` | `2` | `0.0100` |
| `both_gold_in_final` | `48` | `0.2400` |

## 3. Subcategory Counts

| Subcategory | Count | Pct | Recommendation |
| --- | ---: | ---: | --- |
| `budget_limited` | `45` | `0.2250` | increase raw dense depth or add title-aware prefilter before truncating candidates |
| `embedding_confusion` | `2` | `0.0100` | add lexical title prior or hybrid title retrieval before reranking |
| `normalization_or_alias_suspect` | `39` | `0.1950` | tighten title normalization and alias handling before candidate evaluation |
| `query_formulation_gap` | `64` | `0.3200` | prioritize query rewrite or hotpot_decompose instead of retriever-only tuning |
| `rerank_loss` | `2` | `0.0100` | adjust reranker or title packing because both gold titles already survived raw retrieval |
| `resolved` | `48` | `0.2400` | keep as a control group and do not optimize specifically for these samples |

## 4. Top Blockers

1. `query_formulation_gap`: 64 (prioritize query rewrite or hotpot_decompose instead of retriever-only tuning)
2. `budget_limited`: 45 (increase raw dense depth or add title-aware prefilter before truncating candidates)
3. `normalization_or_alias_suspect`: 39 (tighten title normalization and alias handling before candidate evaluation)
4. `embedding_confusion`: 2 (add lexical title prior or hybrid title retrieval before reranking)
5. `rerank_loss`: 2 (adjust reranker or title packing because both gold titles already survived raw retrieval)

## 5. Representative Examples

### `budget_limited`

- `dev_0` Were Scott Derrickson and Ed Wood of the same nationality?
  - gold: ['Scott Derrickson', 'Ed Wood']
  - missing_gold: ['Ed Wood']
  - dense_probe_hits: ['Ed Wood']
  - sparse_probe_hits: ['Ed Wood']
  - alias_candidates: {'Ed Wood': ['Ed Wood', 'Ed Wood (film)', 'Ed Wood (film)', 'Ed Wood (film)', 'Ed Wood (film)']}
- `dev_6` Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?
  - gold: ['Eenasul Fateh', 'Management consulting']
  - missing_gold: ['Eenasul Fateh']
  - dense_probe_hits: ['Eenasul Fateh']
  - sparse_probe_hits: []
  - alias_candidates: {'Eenasul Fateh': ['Eenasul Fateh']}
- `dev_10` What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?
  - gold: ['Kansas Song', 'University of Kansas']
  - missing_gold: ['Kansas Song', 'University of Kansas']
  - dense_probe_hits: ['University of Kansas']
  - sparse_probe_hits: []
  - alias_candidates: {'Kansas Song': ['Kansas (Kansas album)', 'Kansas (Kansas album)', 'Kansas (Kansas album)', 'Kansas (band)', 'Kansas (Kansas album)'], 'University of Kansas': ['History of the University of Kansas', 'University of Kansas', 'University of Kansas', 'Kansas (Kansas album)', 'Kansas (Kansas album)']}
- `dev_12` What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective?
  - gold: ["Oh My God (Guns N' Roses song)", 'End of Days (film)']
  - missing_gold: ["Oh My God (Guns N' Roses song)", 'End of Days (film)']
  - dense_probe_hits: ["Oh My God (Guns N' Roses song)"]
  - sparse_probe_hits: []
  - alias_candidates: {"Oh My God (Guns N' Roses song)": ["Oh My God (Guns N' Roses song)"]}
- `dev_20` Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990?
  - gold: ['Formula One drivers from Mexico', 'Sergio Pérez']
  - missing_gold: ['Formula One drivers from Mexico', 'Sergio Pérez']
  - dense_probe_hits: ['Formula One drivers from Mexico']
  - sparse_probe_hits: []
  - alias_candidates: {'Formula One drivers from Mexico': ['Formula One drivers from Mexico', 'Formula One drivers from Mexico', 'Formula One drivers from Argentina', 'Formula One drivers from Spain', 'Formula One drivers from Argentina']}
- `dev_22` Aside from the Apple Remote, what other device can control the program Apple Remote was originally designed to interact with?
  - gold: ['Apple Remote', 'Front Row (software)']
  - missing_gold: ['Apple Remote', 'Front Row (software)']
  - dense_probe_hits: ['Apple Remote']
  - sparse_probe_hits: ['Apple Remote']
  - alias_candidates: {'Apple Remote': ['Apple Remote', 'Apple Remote', 'Apple Remote', 'Apple Remote', 'Apple Remote']}
- `dev_24` What was the father of Kasper Schmeichel voted to be by the IFFHS in 1992?
  - gold: ['Kasper Schmeichel', 'Peter Schmeichel']
  - missing_gold: ['Kasper Schmeichel']
  - dense_probe_hits: ['Kasper Schmeichel']
  - sparse_probe_hits: ['Kasper Schmeichel']
  - alias_candidates: {'Kasper Schmeichel': ['Kasper Schmeichel', 'Kasper Schmeichel', 'Kasper Schmeichel', 'Kasper Schmeichel', 'Kasper Schmeichel']}
- `dev_26` The 2011–12 VCU Rams men's basketball team, led by third year head coach Shaka Smart, represented Virginia Commonwealth University which was founded in what year?
  - gold: ["2011–12 VCU Rams men's basketball team", 'Virginia Commonwealth University']
  - missing_gold: ["2011–12 VCU Rams men's basketball team"]
  - dense_probe_hits: ["2011–12 VCU Rams men's basketball team"]
  - sparse_probe_hits: ["2011–12 VCU Rams men's basketball team"]
  - alias_candidates: {"2011–12 VCU Rams men's basketball team": ['VCU Rams', "VCU Rams men's basketball", "VCU Rams men's soccer", "2011–12 VCU Rams men's basketball team", "2011–12 VCU Rams men's basketball team"]}
- `dev_45` A medieval fortress in Dirleton, East Lothian, Scotland borders on the south side of what coastal area?
  - gold: ['Yellowcraigs', 'Dirleton Castle']
  - missing_gold: ['Yellowcraigs', 'Dirleton Castle']
  - dense_probe_hits: ['Dirleton Castle']
  - sparse_probe_hits: ['Dirleton Castle']
  - alias_candidates: {'Dirleton Castle': ['Dirleton Castle', 'Dirleton Castle', 'Dirleton Castle', 'Dirleton Castle', 'Dirleton Castle']}
- `dev_55` Which Australian city founded in 1838 contains a boarding school opened by a Prime Minister of Australia and named after a school in London of the same name.
  - gold: ['Westminster School, Adelaide', 'Marion, South Australia']
  - missing_gold: ['Westminster School, Adelaide', 'Marion, South Australia']
  - dense_probe_hits: ['Westminster School, Adelaide']
  - sparse_probe_hits: []
  - alias_candidates: {'Westminster School, Adelaide': ['Westminster School, Adelaide', 'Westminster School, Adelaide']}

### `embedding_confusion`

- `dev_23` Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice? 
  - gold: ['Badly Drawn Boy', 'Wolf Alice']
  - missing_gold: ['Wolf Alice']
  - dense_probe_hits: []
  - sparse_probe_hits: ['Wolf Alice']
  - alias_candidates: {'Wolf Alice': ['Alice Wolf', 'Wolf Alice', 'Alice (spacecraft instrument)', 'Alice Francis Wolf', 'Alice De Wolf Kellogg']}
- `dev_143` Josey Scott and Ian Watkins were both promising musicians. Which of these talented men was incarcerated, impacting his career with a rock band?
  - gold: ['Josey Scott', 'Ian Watkins (Lostprophets)']
  - missing_gold: ['Josey Scott']
  - dense_probe_hits: []
  - sparse_probe_hits: ['Josey Scott']
  - alias_candidates: {'Josey Scott': ['Josey Scott']}

### `normalization_or_alias_suspect`

- `dev_2` What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?
  - gold: ['The Hork-Bajir Chronicles', 'Animorphs']
  - missing_gold: ['The Hork-Bajir Chronicles', 'Animorphs']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'Animorphs': ['Animorphs Chronicles', 'Animorphs (video game)', 'Animorphs (TV series)']}
- `dev_14` The football manager who recruited David Beckham managed Manchester United during what timeframe?
  - gold: ['1995–96 Manchester United F.C. season', 'Alex Ferguson']
  - missing_gold: ['1995–96 Manchester United F.C. season']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'1995–96 Manchester United F.C. season': ['Manchester United F.C.', 'Manchester United F.C.', 'Manchester United F.C.', 'F.C. United of Manchester', 'Manchester United F.C.']}
- `dev_15` Brown State Fishing Lake is in a country that has a population of how many inhabitants ?
  - gold: ['Brown State Fishing Lake', 'Brown County, Kansas']
  - missing_gold: ['Brown State Fishing Lake']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'Brown State Fishing Lake': ['Brown Lake (Stradbroke Island)', 'Fishing Lake', 'Brown Lake (Stradbroke Island)', 'Brown Lake (Stradbroke Island)', 'Fishing Lake']}
- `dev_16` The Vermont Catamounts men's soccer team currently competes in a conference that was formerly known as what from 1988 to 1996?
  - gold: ["Vermont Catamounts men's soccer", 'America East Conference']
  - missing_gold: ["Vermont Catamounts men's soccer"]
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {"Vermont Catamounts men's soccer": ["Vermont Catamounts men's ice hockey", "Vermont Catamounts men's ice hockey", 'Vermont Catamounts', "Vermont Catamounts men's basketball", "2017–18 Vermont Catamounts men's basketball team"]}
- `dev_19` Which writer was from England, Henry Roth or Robert Erskine Childers?
  - gold: ['Henry Roth', 'Robert Erskine Childers']
  - missing_gold: ['Robert Erskine Childers']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'Robert Erskine Childers': ['Erskine Childers (author)', 'Erskine Childers (author)', 'Erskine Childers (author)', 'Erskine Childers (author)', 'Erskine Childers (author)']}
- `dev_29` What is the name for the adventure in "Tunnels and Trolls", a game designed by Ken St. Andre?
  - gold: ['Arena of Khazan', 'Tunnels &amp; Trolls']
  - missing_gold: ['Arena of Khazan', 'Tunnels &amp; Trolls']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'Tunnels &amp; Trolls': ['Tunnels & Trolls', 'Tunnels & Trolls', 'Tunnels & Trolls', 'Tunnels & Trolls', 'Tunnels & Trolls']}
- `dev_30` When was Poison's album "Shut Up, Make Love" released?
  - gold: ['Shut Up, Make Love', 'Poison (American band)']
  - missing_gold: ['Shut Up, Make Love']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'Shut Up, Make Love': ['Shut up', 'Make Up (album)', 'Shut Up (Kelly Osbourne album)', 'Make Love', 'Make Up (EP)']}
- `dev_31` Hayden is a singer-songwriter from Canada, but where does Buck-Tick hail from?
  - gold: ['Hayden (musician)', 'Buck-Tick']
  - missing_gold: ['Buck-Tick']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'Buck-Tick': ['Tick, Tick, Tick (film)', 'Tick, Tick, Tick... (Castle)']}
- `dev_33` Are Freakonomics and In the Realm of the Hackers both American documentaries?
  - gold: ['Freakonomics (film)', 'In the Realm of the Hackers']
  - missing_gold: ['Freakonomics (film)']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'Freakonomics (film)': ['Freakonomics Radio', 'Freakonomics Radio', 'Freakonomics Radio', 'Freakonomics Radio']}
- `dev_36` Seven Brief Lessons on Physics was written by an Italian physicist that has worked in France since what year?
  - gold: ['Seven Brief Lessons on Physics', 'Carlo Rovelli']
  - missing_gold: ['Seven Brief Lessons on Physics']
  - dense_probe_hits: []
  - sparse_probe_hits: []
  - alias_candidates: {'Seven Brief Lessons on Physics': ['Physics (Aristotle)', 'Physics (Aristotle)', 'Physics (Aristotle)']}

### `query_formulation_gap`

- `dev_1` What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
  - gold: ['Kiss and Tell (1945 film)', 'Shirley Temple']
  - missing_gold: ['Shirley Temple']
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
- `dev_25` Who was the writer of These Boots Are Made for Walkin' and who died in 2007?
  - gold: ["These Boots Are Made for Walkin'", 'Lee Hazlewood']
  - missing_gold: ['Lee Hazlewood']
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_27` Are both Dictyosperma, and Huernia described as a genus?
  - gold: ['Dictyosperma', 'Huernia']
  - missing_gold: ['Dictyosperma', 'Huernia']
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_28` Kaiser Ventures corporation was founded by an American industrialist who became known as the father of modern American shipbuilding?
  - gold: ['Kaiser Ventures', 'Henry J. Kaiser']
  - missing_gold: ['Kaiser Ventures']
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_41` Where is the company that Sachin Warrier worked for as a software engineer headquartered? 
  - gold: ['Sachin Warrier', 'Tata Consultancy Services']
  - missing_gold: ['Tata Consultancy Services']
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_42` A Japanese manga series based on a 16 year old high school student Ichitaka Seto, is written and illustrated by someone born in what year?
  - gold: ['Masakazu Katsura', 'I&quot;s']
  - missing_gold: ['Masakazu Katsura', 'I&quot;s']
  - dense_probe_hits: []
  - sparse_probe_hits: []

### `rerank_loss`

- `dev_73` What was the name of the 1996 loose adaptation of William Shakespeare's "Romeo & Juliet" written by James Gunn?
  - gold: ['James Gunn', 'Tromeo and Juliet']
  - missing_gold: []
  - dense_probe_hits: []
  - sparse_probe_hits: []
- `dev_151` Marcus Wayne Garland spent nine seasons with an American professional baseball team that is based in Baltimore, Maryland, and was one of the AMerican League's original eight charter franchises when the league was established in what year?
  - gold: ['Wayne Garland', 'Baltimore Orioles']
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
- `dev_9` Are Local H and For Against both from the United States?
  - gold: ['Local H', 'For Against']
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
- `dev_21` This singer of A Rather Blustery Day also voiced what hedgehog?
  - gold: ['A Rather Blustery Day', 'Jim Cummings']
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
