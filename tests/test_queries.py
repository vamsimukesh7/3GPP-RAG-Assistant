TEST_CASES = [
    {
        "query": "What happens when T311 expires?",
        "expected_status": "SUPPORTED",
        "expected_identifier": "t311",
    },
    {
        "query": "What happens when T999 expires?",
        "expected_status": "UNSUPPORTED",
        "expected_identifier": "t999",
    },
    {
        "query": "What happens when T300 expires?",
        "expected_status": "SUPPORTED",
        "expected_identifier": "t300",
    },
    {
        "query": "What actions does the UE perform upon reception of SIB12?",
        "expected_status": "SUPPORTED",
        "expected_identifier": None,
        "expected_section": "5.2.2.4.13",
    },
    {
        "query": "What happens when T331 expires?",
        "expected_status": "SUPPORTED",
        "expected_identifier": "t331",
    },
    {
        "query": "What happens when T320 expires?",
        "expected_status": "SUPPORTED",
        "expected_identifier": "t320",
    },
    {
        "query": "What is RRCReconfiguration?",
        "expected_status": "SUPPORTED",
        "expected_identifier": "rrcreconfiguration",
    },
    {
        "query": "What happens when T301 expires?",
        "expected_status": "SUPPORTED",
        "expected_identifier": "t301",
    },
    {
        "query": "What happens when T346g expires?",
        "expected_status": "SUPPORTED",
        "expected_identifier": "t346g",
    },
    {
        "query": "What happens when T12345 expires?",
        "expected_status": "UNSUPPORTED",
        "expected_identifier": "t12345",
    },
]