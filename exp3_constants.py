beta_vs_repeats = {
    'hsj_rep': {
        'mnist': {
            # 1: "3072000 4096000 6144000 8192000",
            2: "256000 384000 512000 640000 1024000 2048000 3072000 4096000 6144000 8192000 12288000",
            5: "16000 32000 64000 128000 256000 384000 512000 640000 1024000 2048000 4096000",
            10: "200 500 1000 2000 4000 8000 16000 32000 64000 80000 128000 192000 256000 512000",
            20: "2000 4000 8000 16000 30000 32000 64000 72000 128000 256000",
            50: "2000 4000 8000 16000 32000 64000",
            100: "2000 4000 8000 16000 32000 64000",
            200: "50 100 200 400 500 850 1000 2000 4000 8000 16000 32000",
        }
    },
    # 'hsj_rep_psj_delta': {
    #     'mnist': {
    #         5: "8000 16000 32000 64000",
    #         10: "8000 16000 32000 64000",
    #         20: "8000 16000 32000 64000",
    #         50: "8000 16000 32000 64000",
    #         100: "8000 16000 32000 64000",
    #     }
    # }
}
best_repeat = {
    'hsj_rep': {
        'mnist' : {
            2: ("4096000", "8192000", "12288000"),
            5: ("512000", "1024000", "4096000"),
            10: ("80000", "192000", "512000"),
            20: ("30000", "72000", "256000"),
            50: ("4000", "16000", "64000"),
            100: ("2000", "9000", "64000"),
            200: ("400", "850", "32000"),
        }
    },
    # 'hsj_rep_psj_delta': {}
}