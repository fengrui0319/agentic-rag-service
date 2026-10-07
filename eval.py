from rag import get_knowledge_index, retrieve


TEST_CASES = [
    # api.md
    {
        "query": "How do I check whether the service is running?",
        "expected_source": "api.md",
    },
    {
        "query": "Which endpoint provides streaming chat responses?",
        "expected_source": "api.md",
    },
    {
        "query": "Where is the session ID returned for streaming responses?",
        "expected_source": "api.md",
    },

    # architecture.md
    {
        "query": "How does the agent prevent unlimited tool-call loops?",
        "expected_source": "architecture.md",
    },
    {
        "query": "How does retrieval work internally?",
        "expected_source": "architecture.md",
    },
    {
        "query": "How is conversation history prevented from growing without bound?",
        "expected_source": "architecture.md",
    },
    {
        "query": "What tools can the agent call?",
        "expected_source": "architecture.md",
    },

    # project.md
    {
        "query": "Which language model provider does this project use?",
        "expected_source": "project.md",
    },
    {
        "query": "What capabilities does the current agent support?",
        "expected_source": "project.md",
    },

    # irrelevant / should be rejected
    {
        "query": "Who won the football World Cup?",
        "expected_source": None,
    },
    {
        "query": "What is the weather in Tokyo today?",
        "expected_source": None,
    },
]


def evaluate():
    chunks, chunk_embeddings = get_knowledge_index()

    relevant_total = 0
    hit_at_1 = 0
    hit_at_3 = 0

    rejection_total = 0
    correct_rejections = 0

    for index, case in enumerate(TEST_CASES, start=1):
        results = retrieve(
            case["query"],
            chunks,
            chunk_embeddings,
            top_k=3,
        )

        returned_sources = [
            result["source"]
            for result in results
        ]

        expected_source = case["expected_source"]

        print(f"\nCase {index}")
        print("query:", case["query"])
        print("expected:", expected_source)
        print("returned:", returned_sources)

        if expected_source is None:
            rejection_total += 1

            if not results:
                correct_rejections += 1
                print("result: PASS")
            else:
                print("result: FAIL")

            continue

        relevant_total += 1

        if returned_sources and returned_sources[0] == expected_source:
            hit_at_1 += 1

        if expected_source in returned_sources:
            hit_at_3 += 1

        if expected_source in returned_sources:
            print("result: PASS")
        else:
            print("result: FAIL")

    print("\n==============================")
    print("Evaluation Summary")
    print("==============================")

    if relevant_total:
        print(
            f"Retrieval Hit@1: "
            f"{hit_at_1}/{relevant_total} "
            f"({hit_at_1 / relevant_total:.1%})"
        )

        print(
            f"Retrieval Hit@3: "
            f"{hit_at_3}/{relevant_total} "
            f"({hit_at_3 / relevant_total:.1%})"
        )

    if rejection_total:
        print(
            f"Rejection accuracy: "
            f"{correct_rejections}/{rejection_total} "
            f"({correct_rejections / rejection_total:.1%})"
        )


if __name__ == "__main__":
    evaluate()