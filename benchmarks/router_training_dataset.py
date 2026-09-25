"""
Hand-labeled ground-truth dataset for validating the AI Router's learning loop.

This is intentionally larger and more diverse than the two small synthetic seed
sets already in the codebase (16 examples in app/services/router_exporter.py,
28 in scripts/train_router.py) — those are fine as instant bootstrap seeds, but
too small to produce a meaningful train/eval split or say anything credible
about real accuracy. This one is sized to actually support an 80/20 split with
enough eval samples per class to matter.

Labeling philosophy: the router's actual features (see app/engine/ai_router.py
FEATURE_NAMES) are shallow structural signals — token/char count, turn count,
presence of "complex"/"simple" keywords, code-block signals. Labels here are
assigned based on genuine task complexity as a human would judge it, not
reverse-engineered from the feature list — but low/medium/high complexity does
correlate with prompt length and code presence in practice, so don't expect
these features to be independent of the labels.

Each entry is (messages, label). Most are single-turn; a handful of "high"
entries are multi-turn to exercise turn_count/avg_tokens_per_turn.
"""

from typing import Dict, List, Tuple

Messages = List[Dict[str, str]]


def _u(text: str) -> Messages:
    return [{"role": "user", "content": text}]


LOW_PROMPTS = [
    "What is the capital of Japan?",
    "What is the capital of Brazil?",
    "What is the capital of Canada?",
    "Who wrote Pride and Prejudice?",
    "Who painted the Mona Lisa?",
    "What year did the Berlin Wall fall?",
    "What is the boiling point of water in Fahrenheit?",
    "How many continents are there?",
    "How many centimeters are in 3 feet?",
    "How many ounces are in a pound?",
    "Translate 'thank you' to French.",
    "Translate 'good night' to German.",
    "Translate 'where is the bathroom' to Italian.",
    "Define osmosis in simple terms.",
    "Define inflation in simple terms.",
    "Define a black hole in simple terms.",
    "List 3 primary colors.",
    "List 5 planets in our solar system.",
    "List 3 states of matter.",
    "What is 12 times 8?",
    "What is 144 divided by 12?",
    "What is the square root of 81?",
    "Convert 100 Fahrenheit to Celsius.",
    "Convert 5 kilometers to miles.",
    "Spell the word 'necessary' correctly.",
    "Correct the spelling: 'recieve'.",
    "True or false: the Great Wall of China is visible from space.",
    "Yes or no: is Pluto currently classified as a planet?",
    "What does HTTP stand for?",
    "What does CPU stand for?",
    "What is the chemical symbol for gold?",
    "What is the chemical symbol for sodium?",
    "Name the largest ocean on Earth.",
    "Name the longest river in the world.",
    "What language is primarily spoken in Brazil?",
    "What is the currency used in Japan?",
    "Format this date as YYYY-MM-DD: March 5th, 2024.",
    "Extract the phone number from this text: 'Call me at 555-1234.'",
    "Summarize this sentence in five words: 'The quick brown fox jumps over the lazy dog.'",
    "What is the freezing point of water in Celsius?",
    "How many days are in a leap year?",
    "How many minutes are in a day?",
    "What is the opposite of 'ancient'?",
    "What is a synonym for 'happy'?",
    "Who is the current monarch of the United Kingdom?",
    "What is the tallest mountain in the world?",
    "What color do you get when you mix blue and yellow?",
    "What is the plural of 'cactus'?",
    "Round 3.14159 to two decimal places.",
    "What day of the week was January 1, 2000?",
]

MEDIUM_PROMPTS = [
    "Summarize the key differences between TCP and UDP.",
    "Summarize the causes of the French Revolution in a paragraph.",
    "Explain how photosynthesis works at a high level.",
    "Explain how vaccines train the immune system.",
    "Explain the difference between supervised and unsupervised learning.",
    "Explain the difference between a stack and a queue.",
    "Compare REST and GraphQL APIs for a mobile app backend.",
    "Compare SQL and NoSQL databases for a social media app.",
    "Compare Python and JavaScript for scripting tasks.",
    "How does DNS resolution work when you visit a website?",
    "How does garbage collection work in most modern languages?",
    "How does public-key cryptography enable secure communication?",
    "What are the SOLID principles in object-oriented design?",
    "What are the tradeoffs of microservices versus a monolith?",
    "What are the pros and cons of remote work for a small team?",
    "Write a Python function to check if a string is a palindrome.",
    "Write a Python function to find the maximum value in a list.",
    "Write a SQL query to select the top 5 rows ordered by a date column.",
    "Write a function that reverses a linked list in Java.",
    "Write a regular expression to validate an email address.",
    "Explain how Git branching and merging works with an example.",
    "Explain the CAP theorem in distributed systems.",
    "Explain how a hash table achieves average O(1) lookups.",
    "Describe the steps involved in a typical CI/CD pipeline.",
    "Describe how load balancers distribute traffic across servers.",
    "Outline a study plan for learning a new programming language in a month.",
    "Outline the steps to set up a basic Flask web server.",
    "Analyze the pros and cons of using Docker for local development.",
    "Analyze why normalized database schemas reduce data duplication.",
    "What's the difference between authentication and authorization?",
    "What's the difference between a process and a thread?",
    "What's the difference between deep learning and traditional machine learning?",
    "Explain how HTTPS certificates establish trust between client and server.",
    "Explain why index selectivity matters for database query performance.",
    "Summarize how a binary search algorithm narrows down its search space.",
    "Summarize the main stages of the software development lifecycle.",
    "How would you explain recursion to someone who has never programmed?",
    "How would you decide between a relational and a document database for a new project?",
    "Write a short function in JavaScript to debounce a button click handler.",
    "Write a Python script to read a CSV file and print the column headers.",
    "Explain what a race condition is and give a simple example.",
    "Explain the difference between optimistic and pessimistic UI updates.",
    "Compare unit tests and integration tests, and when to use each.",
    "Compare synchronous and asynchronous programming models.",
    "Describe how OAuth 2.0 authorization code flow works at a high level.",
    "What factors should you consider when choosing a caching strategy?",
    "How does horizontal scaling differ from vertical scaling?",
    "Explain what idempotency means in the context of REST APIs.",
    "Write a function to deduplicate a list while preserving order.",
    "Summarize how content delivery networks (CDNs) reduce latency.",
]

HIGH_PROMPTS: List[Tuple[Messages, str]] = [
    (_u(
        "Design a distributed rate limiter using Redis that works correctly across "
        "multiple servers, handling clock skew and burst traffic. Explain the algorithm "
        "and tradeoffs step by step."
    ), "high"),
    (_u(
        "Architect a real-time chat system that needs to support 1 million concurrent "
        "users, including message ordering, delivery guarantees, and horizontal scaling "
        "of the WebSocket layer."
    ), "high"),
    (_u(
        "Design a URL shortener service that handles 100,000 requests per second, "
        "including ID generation strategy, database sharding, and cache invalidation."
    ), "high"),
    (_u(
        "Implement a thread-safe LRU cache in Python with O(1) get and put operations. "
        "Explain your data structure choice and provide the full implementation with "
        "test cases."
    ), "high"),
    (_u(
        "Implement an async, deadlock-free transaction runner in Python using SQLAlchemy, "
        "including retry backoff logic. ```python\nclass TransactionRunner:\n    pass\n``` "
        "Fill in the implementation and explain the retry strategy."
    ), "high"),
    (_u(
        "Debug this race condition in the following async Python code and rewrite it "
        "to be thread-safe:\n```python\ncounter = 0\nasync def increment():\n    global counter\n"
        "    counter += 1\n```\nExplain exactly why the bug occurs."
    ), "high"),
    (_u(
        "Refactor this code to use the repository pattern and add unit tests:\n"
        "```python\ndef get_user(id):\n    conn = sqlite3.connect('db.sqlite')\n"
        "    return conn.execute('SELECT * FROM users WHERE id=?', (id,)).fetchone()\n```"
    ), "high"),
    (_u(
        "Prove that quicksort has O(n log n) average-case time complexity by deriving "
        "and solving the recurrence relation using the master theorem."
    ), "high"),
    (_u(
        "Prove that a binary search tree's in-order traversal produces sorted output, "
        "using induction on tree height."
    ), "high"),
    (_u(
        "Analyze the security implications of using MD5 for password hashing versus "
        "bcrypt or Argon2, including attack vectors like rainbow tables and GPU cracking."
    ), "high"),
    (_u(
        "Analyze the tradeoffs between eventual consistency and strong consistency for a "
        "globally distributed inventory management system, with concrete failure scenarios."
    ), "high"),
    (_u(
        "Design a multi-region active-active database replication strategy that tolerates "
        "a full region outage with less than 5 seconds of data loss, and explain the "
        "consensus protocol you'd use and why."
    ), "high"),
    (_u(
        "Write a compiler front-end lexer and parser in Rust for a small arithmetic "
        "expression language, including error recovery for malformed input, with full code."
    ), "high"),
    (_u(
        "Implement a consistent hashing ring in Go for a distributed cache, including "
        "virtual nodes to handle uneven key distribution, with full working code."
    ), "high"),
    (_u(
        "Design a fraud detection pipeline for a payments platform processing 10,000 "
        "transactions per second, covering feature engineering, model serving latency "
        "budgets, and how you'd handle concept drift."
    ), "high"),
    (_u(
        "Derive the backpropagation gradient update rule for a two-layer neural network "
        "with a sigmoid activation and mean-squared-error loss, showing all steps."
    ), "high"),
    (_u(
        "Explain and implement the Raft consensus algorithm's leader election process in "
        "pseudocode, covering split-vote scenarios and randomized election timeouts."
    ), "high"),
    (_u(
        "Design a schema migration strategy for a 500GB production PostgreSQL table that "
        "adds a NOT NULL column with zero downtime, covering locking behavior and backfill."
    ), "high"),
    (
        [
            {"role": "system", "content": "You are a senior distributed systems architect."},
            {"role": "user", "content": "We have legacy microservices running across three AWS regions with high latency between them, and we're seeing intermittent network timeouts during peak hours."},
            {"role": "assistant", "content": "That's likely a cross-region connection pooling and retry issue. Can you share more about your current timeout and retry configuration?"},
            {"role": "user", "content": "We use default HTTP client settings with a 30 second timeout and no circuit breaker. What specific timeout thresholds, retry/backoff algorithm, and circuit breaker configuration should we implement for the inter-region RPCs, and how should we test it under simulated network partition?"},
        ],
        "high",
    ),
    (
        [
            {"role": "system", "content": "You are a database performance consultant."},
            {"role": "user", "content": "Our orders table has grown to 200 million rows and queries filtering by customer_id and date range are taking over 10 seconds."},
            {"role": "assistant", "content": "That points to missing or ineffective indexing, and possibly a need for partitioning. What indexes currently exist on the table?"},
            {"role": "user", "content": "Only a primary key on order_id. Design a complete indexing and partitioning strategy for this table, including whether to use range or hash partitioning by date, composite index column order, and how to migrate the existing 200M rows without extended downtime."},
        ],
        "high",
    ),
    (_u(
        "Implement a lock-free multi-producer single-consumer queue in C++ using atomic "
        "operations, and explain the memory ordering guarantees required for correctness."
    ), "high"),
    (_u(
        "Design the sharding strategy for a multiplayer game server handling 500,000 "
        "concurrent players, including how you'd handle players interacting across shard "
        "boundaries in real time."
    ), "high"),
    (_u(
        "Write a full implementation of a red-black tree in Java with insertion, deletion, "
        "and rebalancing, including all rotation cases, with test coverage."
    ), "high"),
    (_u(
        "Explain and formally prove the correctness of the two-phase commit protocol for "
        "distributed transactions, including how it handles coordinator failure."
    ), "high"),
    (_u(
        "Design a real-time fraud scoring system that must return a decision within 50ms "
        "at the p99 latency, covering feature store design, model serving architecture, "
        "and fallback behavior when the model service is unavailable."
    ), "high"),
    (_u(
        "Derive the time and space complexity of the Floyd-Warshall all-pairs shortest "
        "path algorithm, and prove why it correctly handles negative edge weights without "
        "negative cycles."
    ), "high"),
    (_u(
        "Design a multi-tenant SaaS database architecture that balances data isolation, "
        "cost efficiency, and query performance across 10,000 tenants of varying size, "
        "comparing schema-per-tenant, shared-schema, and hybrid approaches."
    ), "high"),
    (_u(
        "Implement a custom memory allocator in C with support for coalescing freed "
        "blocks, and explain the tradeoffs versus using malloc/free directly."
    ), "high"),
    (_u(
        "Analyze the security implications of a JWT-based authentication system that "
        "doesn't verify the 'alg' header, including how an attacker could exploit the "
        "'none' algorithm vulnerability, and design a fix."
    ), "high"),
    (_u(
        "Design an exactly-once event processing pipeline using Kafka, covering "
        "idempotency keys, consumer offset management, and how to handle poison-pill "
        "messages without blocking the partition."
    ), "high"),
    (_u(
        "Write a recursive-descent parser for a JSON-like configuration language in "
        "Python, including proper error messages with line and column numbers for "
        "malformed input."
    ), "high"),
    (_u(
        "Prove that the greedy algorithm for the fractional knapsack problem produces an "
        "optimal solution, and explain why the same greedy approach fails for the 0/1 "
        "knapsack variant."
    ), "high"),
    (_u(
        "Design a globally distributed feature flag system that must propagate flag "
        "changes to all regions within 1 second while remaining available during a "
        "regional network partition."
    ), "high"),
    (_u(
        "Implement a rate-limited, retrying HTTP client wrapper in Go with exponential "
        "backoff and jitter, circuit breaking, and per-host connection pooling, with full "
        "working code."
    ), "high"),
    (_u(
        "Explain how a modern JIT compiler decides which functions to optimize, including "
        "the tradeoffs between tiered compilation and the deoptimization process when "
        "assumptions are invalidated."
    ), "high"),
    (_u(
        "Design a data pipeline that ingests 1TB of clickstream data per day, performs "
        "deduplication and sessionization, and makes the result queryable within 15 "
        "minutes of ingestion, including your choice of storage and processing engines."
    ), "high"),
    (_u(
        "Derive the update rule for the Adam optimizer from first principles, explaining "
        "the role of the first and second moment estimates and bias correction terms."
    ), "high"),
    (_u(
        "Design a zero-downtime blue-green deployment strategy for a stateful service "
        "with an active WebSocket connection pool, covering connection draining and "
        "in-flight request handling during the cutover."
    ), "high"),
    (_u(
        "Write a topological sort implementation in Python that also detects and reports "
        "the specific cycle when the graph is not a DAG, with full code and test cases."
    ), "high"),
    (_u(
        "Analyze the tradeoffs of using vector clocks versus Lamport timestamps for "
        "ordering events in a distributed system, and design a conflict-resolution "
        "strategy for a multi-master replicated key-value store."
    ), "high"),
]


def build_dataset() -> List[Tuple[Messages, str]]:
    """Returns a flat list of (messages, label) tuples across all three tiers."""
    dataset: List[Tuple[Messages, str]] = []
    dataset += [(_u(p), "low") for p in LOW_PROMPTS]
    dataset += [(_u(p), "medium") for p in MEDIUM_PROMPTS]
    dataset += HIGH_PROMPTS
    return dataset


if __name__ == "__main__":
    ds = build_dataset()
    counts = {"low": 0, "medium": 0, "high": 0}
    for _, label in ds:
        counts[label] += 1
    print(f"Total examples: {len(ds)}")
    print(f"By label: {counts}")
