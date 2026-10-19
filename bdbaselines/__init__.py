"""Baseline backdoor detectors for checkpoint auditing, with each method's own decision rule.

    weights        Fields (Dixon's Q), DFBScanner's 62 indicators, row-norm ratio
    probes         HTell (head and LC-1 probes), ProP
    bn             CLP, BNP
    optimization   MM-BD, FreeEagle
    rules          the native decision rules; decide applies them, score computes the scores, validate compares with the
                   released code
"""
